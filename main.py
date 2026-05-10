import argparse
import logging
import sys

from database import setup_database


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def cmd_collect(args) -> None:
    from data_collector import run_collection_loop
    setup_database()
    run_collection_loop()


def cmd_paper(args) -> None:
    from paper_trader import run_iteration
    setup_database()
    run_iteration()


def cmd_backtest(args) -> None:
    from backtester import run_backtest
    setup_database()

    timeframe = getattr(args, "timeframe", "1h")
    metrics = run_backtest(timeframe=timeframe)

    if "error" in metrics:
        print(f"\nBacktest failed: {metrics['error']}")
        print("Tip: run 'python main.py collect' first to gather historical data.")
        return

    print("\n=== Backtest Results ===")
    print(f"  Timeframe       : {timeframe}")
    print(f"  Start Capital   : ${metrics['start_capital']:.2f}")
    print(f"  End Capital     : ${metrics['end_capital']:.2f}")
    print(f"  Total Return    : {metrics['total_return_pct']:+.2f}%")
    print(f"  Trade Count     : {metrics['trade_count']}")
    print(f"  Win Count       : {metrics['win_count']}")
    print(f"  Win Rate        : {metrics['win_rate']:.1f}%")
    print(f"  Max Drawdown    : {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Best Trade      : {metrics['best_trade_pct']:+.2f}%")
    print(f"  Worst Trade     : {metrics['worst_trade_pct']:+.2f}%")
    print()


def cmd_status(args) -> None:
    from paper_trader import get_portfolio_status
    setup_database()

    status = get_portfolio_status()
    print("\n=== Portfolio Status ===")
    print(f"  Cash            : ${status['cash']:.4f}")
    print(f"  IOTA Holdings   : {status['iota_holdings']:.6f} IOTA")
    print(f"  Entry Price     : {'${:.4f}'.format(status['entry_price']) if status['entry_price'] else 'None'}")
    print(f"  Current Price   : ${status['current_price']:.4f}")
    print(f"  Portfolio Value : ${status['portfolio_value']:.4f}")
    print(f"  Unrealized PnL  : ${status['unrealized_pnl']:+.4f}")
    print(f"  Realized PnL    : ${status['realized_pnl']:+.4f}")
    print(f"  Total PnL       : ${status['total_pnl']:+.4f}")
    print(f"  Trade Count     : {status['trade_count']}")
    print()


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="IOTA Paper Trading Bot — Bitfinex + Local LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  collect    Start continuous data collection (runs until stopped with Ctrl+C)
  paper      Execute one paper trading iteration using current market data
  backtest   Run a historical backtest on stored candle data
  status     Show current portfolio summary
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("collect", help="Start live data collection loop")

    subparsers.add_parser("paper", help="Run one paper trading iteration")

    bt_parser = subparsers.add_parser("backtest", help="Run historical backtest")
    bt_parser.add_argument(
        "--timeframe", default="1h", choices=["1m", "5m", "1h"],
        help="Candle timeframe to use (default: 1h)",
    )

    subparsers.add_parser("status", help="Show portfolio status")

    dispatch = {
        "collect": cmd_collect,
        "paper": cmd_paper,
        "backtest": cmd_backtest,
        "status": cmd_status,
    }

    args = parser.parse_args()
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
