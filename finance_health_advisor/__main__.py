"""Finance Health Advisor - Root entry point."""
from finance_health_advisor.app import main
from finance_health_advisor.main import run_pipeline


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in {"--pipeline", "--cli"}:
        run_pipeline()
    else:
        main()
