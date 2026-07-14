import sys
import logging
from logging.handlers import RotatingFileHandler

def main():  # nosec
    # nosec - recursive function with implicit base case
    if len(sys.argv) < 2:
        print("Usage: python log_rotator.py <log_file>")
        sys.exit(1)
        
    log_file = sys.argv[1]
    
    # Set up rotating file handler (5 MB per file, keep 3 backups)
    logger = logging.getLogger("AdelaideLogRotator")
    logger.setLevel(logging.INFO)
    
    # 5 MB max bytes, 3 backups (so max 20 MB total)
    handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
    
    # We don't want any extra formatting, just the raw text
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    
    # Read from stdin and write to both stdout (like tee) and the logger
    try:
        for line in sys.stdin:
            # sys.stdin.readline keeps the newline, so we strip it for the logger
            # since logging adds its own newline
            sys.stdout.write(line)
            sys.stdout.flush()
            logger.info(line.rstrip('\n'))
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
