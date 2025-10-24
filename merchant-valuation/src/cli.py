"""
Command-line interface for the merchant valuation feature pipeline.
"""

import click
from .features import run_pipeline


@click.command()
@click.option(
    '--xlsx',
    type=click.Path(exists=True),
    required=True,
    help='Path to the input Excel file (HackathonDataset.xlsx)'
)
@click.option(
    '--outdir',
    type=click.Path(),
    default='outputs',
    help='Output directory for CSV files (default: outputs)'
)
def main(xlsx: str, outdir: str) -> None:
    """
    Run the merchant valuation feature engineering pipeline.
    
    This script processes hourly generation and price data from Excel,
    computes monthly block statistics, and generates future block templates.
    """
    try:
        run_pipeline(xlsx, outdir)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


if __name__ == '__main__':
    main()
