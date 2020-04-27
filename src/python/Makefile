MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR := $(dir $(MKFILE_PATH))

check-style:
	@echo Running style check for in $(MKFILE_DIR)
	flake8 $(MKFILE_DIR)
	isort -rc $(MKFILE_DIR)
	black $(MKFILE_DIR)
