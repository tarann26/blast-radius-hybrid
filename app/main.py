# Entry point (CMD python /app/main.py): selector.py then loads from its compiled .pyc instead
# of being compiled from source on every start (about 40 ms of CPU).
import selector

selector.main()
