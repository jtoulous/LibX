import logging
import argparse as ap
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt



class Engine():
    def __init__(self, csv_file, agent_file, scaler_file, start, speed, step):
        engine = {}
        self.db = pd.read_csv(csv_file, index_col=False)
        self.db['timestamp'] = pd.to_datetime(self.db['timestamp'])
        self.db = self.db.sort_values(by='timestamp')

        self.agent = Agent(load=True)

        self.start = self.db['timestamp'].iloc[0] if start is None else pd.to_datetime(start)
        self.speed = speed
        self.step= step


    def Replay(self):
        db = self.db.set_index('timestamp')
        start_idx = db.index.get_loc(engine.start)

        window_size = 50
        fig, ax = plt.subplots(figsize=(12, 6))

        for i in range(start_idx, len(db)):
            if not plt.fignum_exists(fig.number):
                break
            ax.clear()
            data = db.iloc[max(0, i - window_size):i+1][['open', 'high', 'low', 'close']]
            breakpoint()
            y_min, y_max = data['low'].min(), data['high'].max()
            y_range = y_max - y_min
            y_padding = y_range * 0.2  

            mpf.plot(data, type='candle', ax=ax, style='charles')
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
            plt.pause(1 / engine.speed)
            if engine.step:
                input("")
        plt.close(fig)


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
        breakpoint()
        engine.Replay()


    except Exception as error:
        print(error)












