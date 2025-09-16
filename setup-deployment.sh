#!/bin/bash
set -e

# SF Business Data Pipeline - Flox Deployment Setup Script
# This script helps set up the complete Flox environment and deployment pipeline

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install Flox if not present
install_flox() {
    if command_exists flox; then
        log_info "Flox is already installed: $(flox --version)"
        return 0
    fi

    log_info "Installing Flox..."

    # Detect architecture and OS
    ARCH=$(uname -m)
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')

    # Install Flox
    curl -sSf "https://downloads.flox.dev/by-env/${ARCH}-${OS}/stable/flox" | sh

    # Add to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"

    if command_exists flox; then
        log_success "Flox installed successfully: $(flox --version)"
    else
        log_error "Failed to install Flox"
        exit 1
    fi
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."

    # Check minimum dependencies
    local missing_deps=()

    if ! command_exists curl; then
        missing_deps+=("curl")
    fi

    if ! command_exists git; then
        missing_deps+=("git")
    fi

    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing required dependencies: ${missing_deps[*]}"
        log_info "Please install them using your system package manager"
        exit 1
    fi

    log_success "System requirements check passed"
}

# Setup Flox environment
setup_environment() {
    log_info "Setting up Flox environment..."

    # Activate Flox environment (this will trigger the hooks)
    flox activate -- echo "Environment activated successfully"

    log_success "Flox environment setup completed"
}

# Verify services can start
verify_services() {
    log_info "Verifying service configurations..."

    # Check if configurations were created
    local config_dirs=(
        ".flox/cache/grafana"
        ".flox/cache/prometheus"
        ".flox/cache/tegola"
        ".flox/cache/nginx"
    )

    for dir in "${config_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            log_warning "Configuration directory missing: $dir"
            log_info "Run 'flox activate' to initialize configurations"
        fi
    done

    # Test database initialization
    log_info "Testing PostgreSQL initialization..."
    flox activate -- bash -c '
        if [ -d "$PGDATA" ]; then
            echo "PostgreSQL data directory exists: $PGDATA"
            if pg_ctl status -D "$PGDATA" >/dev/null 2>&1; then
                echo "PostgreSQL is running"
            else
                echo "PostgreSQL is not running but data directory exists"
            fi
        else
            echo "PostgreSQL will be initialized on first activation"
        fi
    '

    log_success "Service verification completed"
}

# Start services for testing
test_services() {
    log_info "Starting services for testing..."

    # Start PostgreSQL first
    log_info "Starting PostgreSQL..."
    flox activate -- flox services start postgres &
    POSTGRES_PID=$!

    # Wait for PostgreSQL to start
    sleep 5

    # Test PostgreSQL connectivity
    if flox activate -- pg_isready -h localhost -p 5432; then
        log_success "PostgreSQL is running and accepting connections"
    else
        log_error "Failed to connect to PostgreSQL"
        return 1
    fi

    # Test API service (if main.py exists)
    if [ -f "main.py" ]; then
        log_info "Testing API service startup..."
        flox activate -- timeout 10s uv run uvicorn main:app --host 0.0.0.0 --port 2800 &
        API_PID=$!

        sleep 3

        if curl -f http://localhost:2800/health >/dev/null 2>&1; then
            log_success "API service is responding"
            kill $API_PID 2>/dev/null || true
        else
            log_warning "API service health check failed (this may be expected if no health endpoint exists)"
            kill $API_PID 2>/dev/null || true
        fi
    fi

    # Stop test services
    flox activate -- flox services stop

    log_success "Service testing completed"
}

# Setup development environment
setup_development() {
    log_info "Setting up development environment..."

    # Install development dependencies if pyproject.toml exists
    if [ -f "pyproject.toml" ]; then
        log_info "Installing Python dependencies..."
        flox activate -- uv sync
        log_success "Python dependencies installed"
    fi

    # Create sample data directories
    mkdir -p cache output data

    log_success "Development environment setup completed"
}

# Setup container environment
setup_container() {
    log_info "Setting up container environment..."

    # Check if Docker is available
    if command_exists docker; then
        log_info "Docker is available for containerization"

        # Test Flox containerize functionality
        log_info "Testing Flox containerization..."
        if flox activate -- flox build sf-businesses --dry-run >/dev/null 2>&1; then
            log_success "Flox build configuration is valid"
        else
            log_warning "Flox build configuration may need adjustment"
        fi
    else
        log_warning "Docker not found - containerization will not be available"
    fi
}

# Display service information
show_service_info() {
    log_info "Service Information:"
    echo ""
    echo -e "${BLUE}Service URLs (when running):${NC}"
    echo "  API Server:    http://localhost:2800"
    echo "  Web Interface: http://localhost:2801 (nginx)"
    echo "  Vector Tiles:  http://localhost:2802 (tegola)"
    echo "  Prometheus:    http://localhost:2803"
    echo "  Grafana:       http://localhost:2804 (admin/admin)"
    echo ""
    echo -e "${BLUE}Common Commands:${NC}"
    echo "  flox activate                    # Activate environment"
    echo "  flox activate --start-services   # Start all services"
    echo "  flox services status             # Check service status"
    echo "  flox services stop               # Stop all services"
    echo ""
    echo -e "${BLUE}Data Pipeline:${NC}"
    echo "  flox activate -- uv run python refresh.py full    # Run full pipeline"
    echo "  flox activate -- uv run python refresh.py status  # Check status"
    echo ""
}

# Main setup function
main() {
    echo "============================================="
    echo "SF Business Data Pipeline - Flox Setup"
    echo "============================================="
    echo ""

    # Parse command line arguments
    SETUP_TYPE="full"
    SKIP_TESTS=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --development)
                SETUP_TYPE="development"
                shift
                ;;
            --container)
                SETUP_TYPE="container"
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --development    Setup for development only"
                echo "  --container      Setup for container deployment"
                echo "  --skip-tests     Skip service testing"
                echo "  --help           Show this help"
                echo ""
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Run setup steps
    check_requirements
    install_flox
    setup_environment

    case $SETUP_TYPE in
        "development")
            setup_development
            ;;
        "container")
            setup_container
            ;;
        *)
            setup_development
            setup_container
            ;;
    esac

    verify_services

    if [ "$SKIP_TESTS" = false ]; then
        test_services
    fi

    echo ""
    log_success "Setup completed successfully!"
    echo ""
    show_service_info

    echo -e "${GREEN}Next steps:${NC}"
    echo "1. Run 'flox activate --start-services' to start all services"
    echo "2. Open http://localhost:2801 in your browser"
    echo "3. Check the DEPLOYMENT.md file for detailed documentation"
    echo ""
}

# Run main function
main "$@"