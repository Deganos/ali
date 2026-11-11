#!/bin/bash
set -u

echo 'Resolver configuration:'
cat /etc/resolv.conf

for domain in archive.ubuntu.com security.ubuntu.com packages.ros.org; do
    echo "Resolver lookup: $domain"
    timeout 15 getent ahostsv4 "$domain" || echo "Lookup failed or timed out: $domain"
    if command -v dig >/dev/null; then
        echo "Public resolver lookup: $domain"
        timeout 10 dig +time=3 +tries=1 @1.1.1.1 "$domain" A || echo "Public resolver unavailable: $domain"
    fi
done
