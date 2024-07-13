#ifndef MISC_IPADDR_H
#define MISC_IPADDR_H

#include <set>
#include <string>
#include <netinet/in.h>

std::set<std::string> get_ipv4_addr (std::string ifname = "");
bool is_my_ip_address(const std::string ip_addr_or_fqdn);
bool is_valid_ip_or_fqdn(const std::string ip_addr_or_fqdn);
bool is_public_ip_addr(const std::string ip_addr_or_fqdn);
const std::string get_canon_name(const std::string ip_addr);
const std::string get_my_canonical_name();
const std::string get_host_name();
in_addr_t get_my_public_ip();
const std::string addr_ntos(in_addr_t addr);

#endif