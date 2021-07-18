# Create a custom logging level called "METRICS" with a value 60
# 60 is higher than CRITICAL logging. 
# This ensures no other log levels write to file
METRICS = 60

# Unique name for the logger
LOGGER_NAME = 'main'