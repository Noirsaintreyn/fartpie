"""
Thin CLI wrapper around backend.find_options_zones() / backend.fetch_vol_surface()
for standalone testing - the actual logic lives in backend.py (see
find_options_zones docstring there) since it's also used by the
/api/options-zones and /api/vol-surface Flask endpoints, and importing this
script from backend.py would be circular.
"""
import argparse
import contextlib
import io

with contextlib.redirect_stdout(io.StringIO()):
    import backend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instrument', choices=['NQ', 'ES'], required=True)
    ap.add_argument('--exp', required=True, help='expiration date YYYY-MM-DD')
    args = ap.parse_args()

    result = backend.find_options_zones(args.instrument, args.exp)

    print(f"{result['instrument']} via {result['proxy_symbol']}: spot={result['spot']}  dte={result['dte']}  "
          f"atm_iv={result['atm_iv_pct']}%")
    print(f"expected move: IV-implied={result['iv_expected_move_pct']}%  "
          f"GJR-GARCH price-implied={result['price_expected_move_pct']}%")
    if result['regime_note']:
        print(f"regime: {result['regime_note']}")
    for z in sorted(result['zones'], key=lambda z: -z['confluence']):
        print(f"  [{z['source']:<20}] {z['low']} - {z['high']}  (center {z['center']}, "
              f"confluence={z['confluence']}/6)  {z['note']}")


if __name__ == '__main__':
    main()
