# Makefile for AI-Powered Quantitative Research Platform
# Cross-platform build system for Cython/C++ extensions + development tools
#
# Usage:
#   make build          # Build all extensions in-place
#   make clean          # Remove build artifacts
#   make rebuild        # Clean + build
#   make test           # Run tests after build
#   make verify-hash    # Verify build reproducibility
#   make format         # Format code with black
#   make lint           # Lint code with flake8
#
# Supported Platforms: Windows (MSVC), Linux (GCC), macOS (Clang)

# Platform detection
ifeq ($(OS),Windows_NT)
    PLATFORM := Windows
    PYTHON := python
    RM := del /Q
    RMDIR := rmdir /S /Q
    MKDIR := mkdir
    SEP := \\
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Darwin)
        PLATFORM := macOS
    else
        PLATFORM := Linux
    endif
    PYTHON := python3
    RM := rm -f
    RMDIR := rm -rf
    MKDIR := mkdir -p
    SEP := /
endif

# Build directories
BUILD_DIR := build
DIST_DIR := dist
HASH_REPORT := build_hash_report.json

# Colors (for Unix-like systems)
ifndef NO_COLOR
    GREEN := \033[0;32m
    YELLOW := \033[0;33m
    RED := \033[0;31m
    NC := \033[0m
else
    GREEN :=
    YELLOW :=
    RED :=
    NC :=
endif

.PHONY: all build clean rebuild test verify-hash install-build-deps help
.PHONY: format lint no-trade-mask-sample check

# Default target
all: build

help:
	@echo "$(GREEN)AI-Powered Quantitative Research Platform - Build System$(NC)"
	@echo ""
	@echo "$(YELLOW)Build targets:$(NC)"
	@echo "  make build             Build all Cython/C++ extensions in-place"
	@echo "  make clean             Remove build artifacts and compiled extensions"
	@echo "  make rebuild           Clean then build"
	@echo "  make test              Run tests after building"
	@echo "  make verify-hash       Verify build hash report exists"
	@echo "  make install-build-deps Install build dependencies"
	@echo ""
	@echo "$(YELLOW)Development targets:$(NC)"
	@echo "  make format            Format Python code with black"
	@echo "  make lint              Lint code with flake8"
	@echo "  make check             Quick syntax check (no build)"
	@echo "  make no-trade-mask-sample Run no-trade mask sample"
	@echo ""
	@echo "$(YELLOW)Platform:$(NC) $(PLATFORM)"
	@echo "$(YELLOW)Python:$(NC) $(shell $(PYTHON) --version)"

# Install build dependencies
install-build-deps:
	@echo "$(GREEN)Installing build dependencies...$(NC)"
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-build.txt
	@echo "$(GREEN)Build dependencies installed.$(NC)"

# Build all extensions
build:
	@echo "$(GREEN)Building native extensions for $(PLATFORM)...$(NC)"
	@echo "$(YELLOW)This may take a few minutes on first build.$(NC)"
	$(PYTHON) setup.py build_ext --inplace
	@echo ""
	@echo "$(GREEN)✓ Build complete!$(NC)"
	@echo "$(YELLOW)Hash report:$(NC) $(HASH_REPORT)"

# Clean build artifacts
clean:
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
ifeq ($(PLATFORM),Windows)
	-@if exist $(BUILD_DIR) $(RMDIR) $(BUILD_DIR) 2>nul
	-@if exist $(DIST_DIR) $(RMDIR) $(DIST_DIR) 2>nul
	-@if exist $(HASH_REPORT) $(RM) $(HASH_REPORT) 2>nul
	-@for %%f in (*.c) do @$(RM) "%%f" 2>nul
	-@for %%f in (*.pyd) do @$(RM) "%%f" 2>nul
	-@for %%f in (*.so) do @$(RM) "%%f" 2>nul
	-@for %%f in (*.html) do @$(RM) "%%f" 2>nul
else
	$(RMDIR) $(BUILD_DIR) $(DIST_DIR) 2>/dev/null || true
	$(RM) $(HASH_REPORT) 2>/dev/null || true
	find . -maxdepth 1 -name "*.c" -type f -delete 2>/dev/null || true
	find . -maxdepth 1 -name "*.so" -type f -delete 2>/dev/null || true
	find . -maxdepth 1 -name "*.pyd" -type f -delete 2>/dev/null || true
	find . -maxdepth 1 -name "*.html" -type f -delete 2>/dev/null || true
endif
	@echo "$(GREEN)✓ Clean complete.$(NC)"

# Rebuild from scratch
rebuild: clean build

# Run tests (requires pytest)
test: build
	@echo "$(GREEN)Running tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v --tb=short
	@echo "$(GREEN)✓ Tests complete.$(NC)"

# Verify hash report exists
verify-hash:
	@echo "$(YELLOW)Verifying build hash report...$(NC)"
ifeq ($(PLATFORM),Windows)
	@if exist $(HASH_REPORT) (echo $(GREEN)✓ Hash report found: $(HASH_REPORT)$(NC)) else (echo $(RED)✗ Hash report not found. Run 'make build' first.$(NC) && exit 1)
else
	@if [ -f "$(HASH_REPORT)" ]; then \
		echo "$(GREEN)✓ Hash report found: $(HASH_REPORT)$(NC)"; \
		cat $(HASH_REPORT) | $(PYTHON) -m json.tool; \
	else \
		echo "$(RED)✗ Hash report not found. Run 'make build' first.$(NC)"; \
		exit 1; \
	fi
endif

# Quick syntax check (no build)
check:
	@echo "$(YELLOW)Running syntax checks...$(NC)"
	$(PYTHON) -m py_compile setup.py
	@echo "$(GREEN)✓ Syntax OK.$(NC)"

# ============================================================================
# Development Targets (existing)
# ============================================================================

format:
	@echo "$(GREEN)Formatting code with black...$(NC)"
	$(PYTHON) -m black .
	@echo "$(GREEN)✓ Format complete.$(NC)"

lint:
	@echo "$(GREEN)Linting code with flake8...$(NC)"
	$(PYTHON) -m flake8 --max-line-length=200 config.py transformers.py tune_threshold.py update_and_infer.py utils_time.py validate_processed.py watchdog_vec_env.py
	@echo "$(GREEN)✓ Lint complete.$(NC)"

no-trade-mask-sample:
	@echo "$(GREEN)Running no-trade mask sample...$(NC)"
	$(PYTHON) tests/run_no_trade_mask_sample.py
