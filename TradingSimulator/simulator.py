import logging
import argparse as ap
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import pickle

from utils.agent import Agent

class Engine():
    def __init__(self, csv_file, agent_file, scaler_file, start, speed, step):
        engine = {}
        self.db = pd.read_csv(csv_file, index_col=False)
        self.db['timestamp'] = pd.to_datetime(self.db['timestamp'])
        self.db = self.db.sort_values(by='timestamp')

        self.agent = Agent(['RSI14', 'U-BAND', 'L-BAND', 'VOLATILITY10'], load_file=agent_file, scaler_file=scaler_file)

        self.start = self.db['timestamp'].iloc[0] if start is None else pd.to_datetime(start)
        self.speed = speed
        self.step= step


    def Replay(self):
        db = self.db.set_index('timestamp')
        start_idx = db.index.get_loc(engine.start)
        window_size = 50
        trade_lifespan = 0
        sl = 0
        tp = 0
        trade_open = 0
        fig, ax = plt.subplots(figsize=(18, 9))


        for i in range(start_idx, len(db)):
            if not plt.fignum_exists(fig.number):
                break
            ax.clear()

            trade_lifespan, trade_open, sl, tp = self.CheckHit(db.iloc[i], trade_lifespan, trade_open, sl, tp) 
            window_data = db.iloc[max(0, i - window_size):i+1][['open', 'high', 'low', 'close']]
            pred = self.agent.predict(db.iloc[[i]])
            pred = pred[0]

            if pred == 0 and trade_lifespan == 0:
                trade_lifespan = 10
                sl = db.iloc[i]['STOP_LOSS']
                tp = db.iloc[i]['TAKE_PROFIT']
                trade_open = db.iloc[i]['close']

            if trade_lifespan != 0:
                ax.axhline(y=tp, color='green', linestyle='-', label='Take Profit')
                ax.axhline(y=trade_open, color='black', linestyle='-', label='Trade Open')
                ax.axhline(y=sl, color='red', linestyle='-', label='Stop Loss')
            
                ax.axhspan(trade_open, tp, facecolor='green', alpha=0.2)
                ax.axhspan(sl, trade_open, facecolor='red', alpha=0.2)


            mpf.plot(window_data, type='candle', ax=ax, style='charles')
            plt.pause(1 / engine.speed)

            if trade_lifespan != 0:
                trade_lifespan -= 1

            if engine.step:
                input("")
        
        plt.close(fig)


    def CheckHit(self, new_data, trade_lifespan, trade_open, sl, tp):
        if sl >= new_data['low'] or tp <= new_data['high']:
            return 0, 0, 0, 0
        else:
            return trade_lifespan, trade_open, sl, tp




def Parsing():
    parser = ap.ArgumentParser()
    parser.add_argument('csv_file', type=str, help='csv database file, "," as separator')
    parser.add_argument('agent_file', type=str, help='saved agent file .pkl')
    parser.add_argument('scaler_file', type=str, help='saved scaler file .pkl')
    parser.add_argument('-start', type=str, default=None, help='start date for backtest')
    parser.add_argument('-speed', type=float, default=5, help='speed of backtest, 69 for instant results')
    parser.add_argument('-step', action='store_true', default=False, help='step by step mode')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    try:
        args = Parsing()
        engine = Engine(args.csv_file, args.agent_file, args.scaler_file, args.start, args.speed, args.step)
        engine.Replay()


    except Exception as error:
        print(error)












