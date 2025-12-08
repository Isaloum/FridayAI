# Security Policy

## 🔒 Security Commitment

The FridayAI project takes security seriously. We appreciate your efforts to responsibly disclose your findings and will make every effort to acknowledge your contributions.

## 🛡️ Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## 🚨 Reporting a Vulnerability

### How to Report

**DO NOT** open public issues for security vulnerabilities.

Instead, please report security vulnerabilities by emailing:
- **Email**: [your-security-email@example.com]
- **Subject**: [SECURITY] Brief description of the issue

### What to Include

Please include the following information:
- Type of vulnerability
- Full paths of source files related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 7 days
- **Fix Development**: Depends on complexity
- **Patch Release**: As soon as safely possible
- **Public Disclosure**: After patch is released (coordinated disclosure)

## 🔐 Security Best Practices for Users

### 1. Environment Variables

**NEVER** commit sensitive data to version control:

```bash
# ❌ BAD - Never do this
OPENAI_API_KEY = "sk-proj-actual-key-here"

# ✅ GOOD - Use environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

### 2. Credentials Management

- Use `.env` files for local development (NEVER commit these)
- Use secret management services for production (AWS Secrets Manager, Azure Key Vault, etc.)
- Rotate API keys and passwords regularly (every 90 days minimum)
- Use different credentials for development, staging, and production

### 3. .gitignore Configuration

Ensure your `.gitignore` includes:

```gitignore
# Environment files
.env
.env.local
.env.*.local

# Security files
*.pem
*.key
*.enc
*token*.txt
*.p12
*.pfx

# Database files with sensitive data
*.db
*.sqlite
*.sqlite3
```

### 4. API Key Security

**OpenAI API Keys:**
- Store in environment variables only
- Never log or print API keys
- Use separate keys for development and production
- Monitor usage for unexpected patterns
- Set usage limits on your OpenAI account

**GitHub Tokens:**
- Use fine-grained personal access tokens with minimum required permissions
- Rotate tokens regularly
- Revoke unused tokens immediately

### 5. Database Security

- Use strong passwords (minimum 16 characters, mix of upper/lower/numbers/symbols)
- Enable SSL/TLS for database connections
- Restrict database access by IP address
- Use separate database users with limited permissions for applications
- Regular backups with encryption

### 6. Dependency Security

```bash
# Check for known vulnerabilities
pip install safety
safety check

# Keep dependencies updated
pip list --outdated
pip install --upgrade package-name
```

## 🔍 Security Checklist for Contributors

Before submitting code:

- [ ] No hardcoded credentials, API keys, or passwords
- [ ] All secrets loaded from environment variables
- [ ] No sensitive data in comments or debug statements
- [ ] Input validation implemented for user inputs
- [ ] SQL injection prevention (use parameterized queries)
- [ ] XSS prevention (sanitize user inputs)
- [ ] CSRF protection (for web interfaces)
- [ ] Rate limiting implemented for API endpoints
- [ ] Error messages don't expose sensitive information
- [ ] Logging doesn't include sensitive data
- [ ] Dependencies are up-to-date and secure

## 🛠️ Security Features in FridayAI

### Current Security Measures

1. **Environment-based Configuration**
   - All sensitive data loaded from `.env` file
   - `.env` excluded from version control

2. **Encryption Support**
   - Memory encryption using Fernet (cryptography library)
   - Encrypted storage for sensitive user data

3. **Input Sanitization**
   - User input validation and sanitization
   - Protection against injection attacks

4. **API Security**
   - CORS configuration for API endpoints
   - Request validation and error handling

### Planned Security Enhancements

- [ ] Rate limiting for API endpoints
- [ ] Authentication and authorization system
- [ ] Session management with secure tokens
- [ ] Audit logging for sensitive operations
- [ ] Automated security scanning in CI/CD
- [ ] Regular dependency vulnerability scans

## 🚫 Known Security Limitations

### Current Limitations

1. **No Built-in Authentication**
   - API endpoints are currently open
   - Users should implement authentication layer for production use

2. **Local Database Storage**
   - Default SQLite database is not encrypted
   - Consider using encrypted storage for production

3. **No Rate Limiting**
   - API endpoints don't have built-in rate limiting
   - Implement reverse proxy with rate limiting for production

### Recommended for Production

If deploying FridayAI in production:

1. **Add Authentication Layer**
   ```python
   # Use Flask-Login, JWT, or OAuth2
   from flask_jwt_extended import JWTManager, jwt_required
   ```

2. **Enable HTTPS**
   - Use SSL/TLS certificates
   - Redirect HTTP to HTTPS

3. **Implement Rate Limiting**
   ```python
   from flask_limiter import Limiter
   limiter = Limiter(app, key_func=get_remote_address)
   ```

4. **Use Production Database**
   - PostgreSQL with SSL enabled
   - Regular backups
   - Restricted network access

5. **Add Monitoring**
   - Log all security events
   - Monitor for suspicious activity
   - Set up alerts for anomalies

## 📚 Security Resources

### For Developers

- [OWASP Top Ten](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [Flask Security](https://flask.palletsprojects.com/en/latest/security/)
- [API Security Checklist](https://github.com/shieldfy/API-Security-Checklist)

### For Users

- [OpenAI API Best Practices](https://platform.openai.com/docs/guides/safety-best-practices)
- [How to Store API Keys Securely](https://12factor.net/config)
- [Database Security Checklist](https://www.postgresql.org/docs/current/security.html)

## 🏆 Security Hall of Fame

We recognize security researchers who responsibly disclose vulnerabilities:

| Name | Date | Vulnerability Type | Severity |
|------|------|-------------------|----------|
| *Coming soon* | - | - | - |

## 📞 Contact

For security concerns:
- **Security Email**: [your-security-email@example.com]
- **PGP Key**: [Optional: Link to PGP key]

For general questions:
- **GitHub Issues**: [Regular, non-security issues](https://github.com/Isaloum/FridayAI/issues)
- **GitHub Discussions**: [General discussions](https://github.com/Isaloum/FridayAI/discussions)

---

## ⚖️ Disclosure Policy

We follow [Coordinated Vulnerability Disclosure](https://vuls.cert.org/confluence/display/CVD):

1. Security researcher reports vulnerability privately
2. We acknowledge receipt and begin investigation
3. We develop and test a fix
4. We release a patch
5. We coordinate public disclosure with the researcher

## 📝 Security Updates

Security updates will be announced via:
- GitHub Security Advisories
- Release notes
- Project README

Subscribe to releases to stay informed about security updates.

---

**Last Updated**: December 8, 2025  
**Version**: 1.0

*Thank you for helping keep FridayAI and its users safe!* 🙏
