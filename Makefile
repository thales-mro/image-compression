# COMMANDS
MKDIR=mkdir -p
CP=cp
PYTHON=python3.7
RM=rm -r -f

# FOLDERS
SRC_FOLDER=src/
BIN_FOLDER=bin/
INPUT_FOLDER=input/
OUTPUT_FOLDER=output/

# SRCs and OBJs
SRC = $(wildcard $(SRC_FOLDER)/*.py)
MAIN=$(BIN_FOLDER)/main.py

# BUILD COMMAND
build:
	$(MKDIR)  $(BIN_FOLDER)
	$(CP) $(SRC) $(BIN_FOLDER)

# RUN COMMAND
exec:
	$(MKDIR) $(OUTPUT_FOLDER)
	$(PYTHON) $(MAIN)

# CLEAN COMMAND
clean:
	$(RM) $(BIN_FOLDER)
	$(RM) $(OUTPUT_FOLDER)
