# SF Business Data Pipeline - Flox Deployment Guide

## Overview

This project uses Flox for environment management and deployment, providing a complete infrastructure-as-code solution with service orchestration, containerization, and CI/CD pipeline.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Flox Environment                         │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│ │   API       │ │   Nginx     │ │   Tegola    │            │
│ │ Port 2800   │ │ Port 2801   │ │ Port 2802   │            │
│ └─────────────┘ └─────────────┘ └─────────────┘            │
│                                                             │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│ │ Prometheus  │ │   Grafana   │ │ PostgreSQL  │            │
│ │ Port 2803   │ │ Port 2804   │ │ Port 5432   │            │
│ └─────────────┘ └─────────────┘ └─────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

## Services

| Service | Port | Purpose | Health Check |
|---------|------|---------|--------------|
| API Server | 2800 | FastAPI application | `/health` |
| Nginx | 2801 | Reverse proxy & static files | HTTP 200 |
| Tegola | 2802 | Vector tile server | `/capabilities` |
| Prometheus | 2803 | Metrics collection | `/metrics` |
| Grafana | 2804 | Monitoring dashboard | `/api/health` |
| PostgreSQL | 5432 | Database with PostGIS | `pg_isready` |

## Quick Start

### Prerequisites

1. Install Flox:
   ```bash
   curl -sSf https://downloads.flox.dev/by-env/$(uname -m)-$(uname -s | tr '[:upper:]' '[:lower:]')/stable/flox | sh
   ```

2. Clone repository:
   ```bash
   git clone <repository-url>
   cd sf_businesses
   ```

### Local Development

1. **Activate Flox environment:**
   ```bash
   flox activate
   ```
   This will automatically:
   - Set up Python 3.13 virtual environment
   - Install all dependencies via uv
   - Initialize PostgreSQL database
   - Create service configuration files

2. **Start all services:**
   ```bash
   flox activate --start-services
   ```
   Or start services individually:
   ```bash
   flox services start postgres
   flox services start api
   flox services start nginx
   # etc.
   ```

3. **Access services:**
   - **API**: http://localhost:2800
   - **Web Interface**: http://localhost:2801
   - **Vector Tiles**: http://localhost:2802
   - **Prometheus**: http://localhost:2803
   - **Grafana**: http://localhost:2804 (admin/admin)

4. **Run data pipeline:**
   ```bash
   flox activate -- uv run python refresh.py full
   ```

### Service Management

```bash
# Check service status
flox services status

# Start specific service
flox services start postgres

# Stop all services
flox services stop

# Restart a service
flox services restart api

# View service logs (if available)
flox services logs api
```

## Deployment Strategies

### 1. Container Deployment

**Build container:**
```bash
flox activate
flox build sf-businesses
flox containerize --name sf-businesses-app
```

**Run container:**
```bash
docker run -d --name sf-businesses \
  -p 2800-2804:2800-2804 \
  -v sf_data:/data \
  sf-businesses-app:latest
```

**Docker Compose:**
```yaml
# Use the generated docker-compose.staging.yml
docker-compose -f docker-compose.staging.yml up -d
```

### 2. Kubernetes Deployment

**Apply manifests:**
```bash
kubectl apply -f k8s-deployment.yml
```

**Check deployment:**
```bash
kubectl get pods -l app=sf-businesses
kubectl get services
```

### 3. Direct Flox Deployment

**On target server:**
```bash
# Install Flox
curl -sSf https://downloads.flox.dev/by-env/$(uname -m)-$(uname -s | tr '[:upper:]' '[:lower:]')/stable/flox | sh

# Clone and activate
git clone <repository-url>
cd sf_businesses
flox activate --start-services
```

## CI/CD Pipeline

The project includes a GitHub Actions workflow (`.github/workflows/flox-deployment.yml`) that:

1. **Tests** in Flox environment
2. **Builds** container images using `flox containerize`
3. **Deploys** to staging automatically
4. **Deploys** to production on manual trigger

### Pipeline Triggers

- **Push to main**: Automatic staging deployment
- **Pull request**: Test only
- **Manual dispatch**: Choose staging or production

### Environment Variables

Required for CI/CD:
- `GITHUB_TOKEN`: Automatic (for container registry)

Optional:
- `KUBECONFIG`: For Kubernetes deployments
- `DOCKER_REGISTRY`: Custom container registry

## Configuration

### Environment Variables

All configuration is handled through Flox environment variables in `manifest.toml`:

```toml
[vars]
# Database
PGHOST = "localhost"
PGPORT = "5432"
PGUSER = "postgres"
PGDATABASE = "sf_businesses"

# Service Ports
API_PORT = "2800"
NGINX_PORT = "2801"
TEGOLA_PORT = "2802"
PROMETHEUS_PORT = "2803"
GRAFANA_PORT = "2804"

# Data Directories
GRAFANA_DATA_DIR = "$FLOX_ENV_CACHE/grafana"
PROMETHEUS_DATA_DIR = "$FLOX_ENV_CACHE/prometheus"
TEGOLA_CONFIG_DIR = "$FLOX_ENV_CACHE/tegola"
NGINX_CONFIG_DIR = "$FLOX_ENV_CACHE/nginx"
```

### Service Configuration

Service configurations are automatically generated on first activation:

- **Nginx**: Reverse proxy with static file serving
- **Tegola**: PostGIS vector tile server
- **Prometheus**: Metrics collection from all services
- **Grafana**: Pre-configured dashboards and data sources

## Monitoring & Observability

### Built-in Monitoring

1. **Prometheus Metrics**:
   - Application metrics at `/metrics`
   - Service health and performance
   - Custom business metrics

2. **Grafana Dashboards**:
   - System overview
   - API performance
   - Data pipeline health
   - Business metrics visualization

3. **Health Checks**:
   ```bash
   # API health
   curl http://localhost:2800/health

   # Database connectivity
   pg_isready -h localhost -p 5432

   # All services
   flox activate -- /path/to/health-check
   ```

### Log Management

Logs are available in:
- **Application logs**: `cache/refresh_logs/`
- **Service logs**: Via `flox services logs <service>`
- **Nginx logs**: `$NGINX_CONFIG_DIR/access.log`

## Security Considerations

1. **Network Security**:
   - Services bound to localhost by default
   - Use reverse proxy (nginx) for external access
   - Consider firewall rules for production

2. **Database Security**:
   - Local trust authentication for development
   - Configure proper authentication for production
   - Regular backups and encryption

3. **Container Security**:
   - Images built with minimal dependencies
   - Non-root user execution where possible
   - Regular security updates via Flox

## Troubleshooting

### Common Issues

1. **Port conflicts**:
   ```bash
   # Check port usage
   netstat -tulpn | grep 280[0-4]

   # Change ports in manifest.toml if needed
   ```

2. **Service startup failures**:
   ```bash
   # Check service status
   flox services status

   # View logs
   flox services logs <service-name>

   # Restart services
   flox services restart <service-name>
   ```

3. **Database connection issues**:
   ```bash
   # Check PostgreSQL status
   pg_isready -h localhost -p 5432

   # Reinitialize if needed
   rm -rf $FLOX_ENV_CACHE/pgdata
   flox activate  # Will reinitialize
   ```

4. **Container build issues**:
   ```bash
   # Clean and rebuild
   flox build sf-businesses --rebuild

   # Check build logs
   flox build sf-businesses --verbose
   ```

### Performance Tuning

1. **Database**:
   - Adjust `postgresql.conf` for your workload
   - Configure connection pooling
   - Monitor query performance

2. **Application**:
   - Use `--workers` with uvicorn for production
   - Configure async settings
   - Enable caching where appropriate

3. **Monitoring**:
   - Adjust Prometheus scrape intervals
   - Configure retention policies
   - Set up alerting rules

## Development vs Production

### Development
- All services on localhost
- Local file storage
- Debug logging enabled
- Hot reload for API

### Production
- Load balancer/reverse proxy
- Persistent storage volumes
- Structured logging
- Multiple replicas
- Backup strategies
- Security hardening

## Backup and Recovery

### Database Backups
```bash
# Manual backup
pg_dump -h localhost -p 5432 -U postgres sf_businesses > backup.sql

# Automated backups (add to cron)
flox activate -- pg_dump sf_businesses | gzip > "backup_$(date +%Y%m%d_%H%M%S).sql.gz"
```

### Configuration Backups
```bash
# Backup Flox environment
tar -czf flox_config_backup.tar.gz .flox/

# Backup service configurations
tar -czf service_configs.tar.gz $FLOX_ENV_CACHE/{grafana,prometheus,tegola,nginx}
```

## Support

For issues and questions:
1. Check this documentation
2. Review GitHub Issues
3. Check CI/CD pipeline logs
4. Contact development team

---

## Appendix: Flox Commands Reference

```bash
# Environment management
flox activate                    # Activate environment
flox activate --start-services   # Activate and start all services
flox deactivate                 # Deactivate environment

# Service management
flox services start             # Start all services
flox services stop              # Stop all services
flox services status            # Check service status
flox services restart <name>    # Restart specific service

# Package management
flox install <package>          # Install new package
flox search <term>             # Search for packages
flox show <package>            # Show package details

# Build and deployment
flox build                     # Build packages
flox containerize             # Create container image
flox publish                  # Publish to registry
```