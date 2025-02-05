import logging
import argparse as ap
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

def InitEngine(args):
    engine = {}
    engine['db'] = pd.read_csv(args.db_path, index_col=False)
    engine['db']['timestamp'] = pd.to_datetime(engine['db']['timestamp'])
    engine['db'] = engine['db'].sort_values(by='timestamp')

    engine['start'] = engine['db']['timestamp'].iloc[0] if args.start is None else pd.to_datetime(args.start)
    engine['speed'] = args.speed
    engine['step'] = args.step

    return engine

def Parsing():
    parser = ap.ArgumentParser()
    parser.add_argument('db_path', type=str, help='csv database file, "," as separator')
    parser.add_argument('-start', type=str, default=None, help='start date for backtest')
    parser.add_argument('-speed', type=float, default=5, help='speed of backtest, 69 for instant results')
    parser.add_argument('-step', action='store_true', default=False, help='step by step mode')

    args = parser.parse_args()
    return InitEngine(args)

if __name__ == '__main__':
    try:
        engine = Parsing()

        db = engine['db'].set_index('timestamp')
        start_idx = db.index.get_loc(engine['start'])

        window_size = 50
        fig, ax = plt.subplots(figsize=(12, 6))

        for i in range(start_idx, len(db)):
            ax.clear()

            data = db.iloc[max(0, i - window_size):i+1][['open', 'high', 'low', 'close']]

            y_min, y_max = data['low'].min(), data['high'].max()
            y_range = y_max - y_min
            y_padding = y_range * 0.2  

            mpf.plot(data, type='candle', ax=ax, style='charles')

            ax.set_ylim(y_min - y_padding, y_max + y_padding)

            plt.pause(1 / engine['speed'])

            if engine['step']:
                input("")

        plt.show()


    except Exception as error:
        print(error)












