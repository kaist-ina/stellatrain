#include "ipaddr.h"
#include <iostream>
#include <sys/types.h>
#include <ifaddrs.h>
#include <netinet/in.h> 
#include <arpa/inet.h>
#include <netdb.h>
#include <array>
#include <cstring>
#include <unistd.h>

/* https://stackoverflow.com/questions/212528/how-can-i-get-the-ip-address-of-a-linux-machine */

std::set<std::string> get_ipv4_addr (std::string ifname) {
    struct ifaddrs * ifAddrStruct = nullptr;
    struct ifaddrs * ifa = nullptr;
    std::set<std::string> result;

    getifaddrs(&ifAddrStruct);

    for (ifa = ifAddrStruct; ifa != nullptr; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr) {
            continue;
        }
        if (ifa->ifa_addr->sa_family == AF_INET) { // check it is IP4
            // is a valid IP4 Address
            auto tmpAddrPtr=&((struct sockaddr_in *)ifa->ifa_addr)->sin_addr;
            char addressBuffer[INET_ADDRSTRLEN] = {0, };
            inet_ntop(AF_INET, tmpAddrPtr, addressBuffer, INET_ADDRSTRLEN);
            if (ifname.length() == 0 || std::string(ifa->ifa_name) == ifname) {
                result.insert(std::string(addressBuffer));
            }
        }
    }
    if (ifAddrStruct!=nullptr)
        freeifaddrs(ifAddrStruct);

    return result;
}

void* getSinAddr(addrinfo *addr)
{
    switch (addr->ai_family)
    {
        case AF_INET:
            return &(reinterpret_cast<sockaddr_in*>(addr->ai_addr)->sin_addr);

        case AF_INET6:
            return &(reinterpret_cast<sockaddr_in6*>(addr->ai_addr)->sin6_addr);
    }

    return NULL;
}

bool is_valid_ip_or_fqdn(const std::string ip_addr_or_fqdn) {
    struct addrinfo hints;
    hints.ai_flags = AI_CANONNAME;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    struct addrinfo *res;
    int ret = getaddrinfo(ip_addr_or_fqdn.c_str(), NULL, &hints, &res);
    if (ret != 0) {
        return false;
    }
    return true;
}

bool is_my_ip_address(const std::string ip_addr_or_fqdn) {
    struct addrinfo hints;
    hints.ai_flags = AI_CANONNAME;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    auto set_addr = get_ipv4_addr();

    if (set_addr.find(ip_addr_or_fqdn) != set_addr.end()) {
        return true;
    }

    struct addrinfo *res;
    int ret = getaddrinfo(ip_addr_or_fqdn.c_str(), NULL, &hints, &res);
    if (ret != 0) {
        std::cerr << "getaddrinfo() failed: " << gai_strerror(ret) << std::endl;
        return false;
    }

    std::cout << res->ai_canonname << "\n";

    char ip[INET6_ADDRSTRLEN];
    bool found = false;
    for(addrinfo *addr = res; addr; addr = addr->ai_next) {
        std::string ip_addr(inet_ntop(addr->ai_family, getSinAddr(addr), ip, sizeof(ip)));
        if (set_addr.find(ip_addr) != set_addr.end()) {
            found = true;
            break;
        }
    }

    freeaddrinfo(res);
    return found;
}



static bool is_public_ip_addr_(const std::string ip_addr) {
    const std::array<std::pair<in_addr_t, in_addr_t>, 9> public_ip_ranges = {
        std::make_pair(inet_addr("1.0.0.0"), inet_addr("9.255.255.255")),
        std::make_pair(inet_addr("11.0.0.0"), inet_addr("126.255.255.255")),
        std::make_pair(inet_addr("129.0.0.0"), inet_addr("169.253.255.255")),
        std::make_pair(inet_addr("169.255.0.0"), inet_addr("172.15.255.255")),
        std::make_pair(inet_addr("172.32.0.0"), inet_addr("191.0.1.255")),
        std::make_pair(inet_addr("192.0.3.0"), inet_addr("192.88.98.255")),
        std::make_pair(inet_addr("192.88.100.0"), inet_addr("192.167.255.255")),
        std::make_pair(inet_addr("192.169.0.0"), inet_addr("198.17.255.255")),
        std::make_pair(inet_addr("198.20.0.0"), inet_addr("223.255.255.255")),
    };

    const auto target = ntohl(inet_addr(ip_addr.c_str()));

    for (auto it = public_ip_ranges.begin(); it != public_ip_ranges.end(); ++it) {
        if (target >= ntohl(it->first) && target <= ntohl(it->second)) {
            return true;
        }
    }
    return false;
}

bool is_public_ip_addr(const std::string ip_addr_or_fqdn) {
    struct addrinfo hints;
    hints.ai_flags = AI_CANONNAME;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    auto set_addr = get_ipv4_addr();

    struct addrinfo *res;
    int ret = getaddrinfo(ip_addr_or_fqdn.c_str(), NULL, &hints, &res);
    if (ret != 0) {
        std::cerr << "getaddrinfo() failed: " << gai_strerror(ret) << std::endl;
        return false;
    }

    char ip[INET6_ADDRSTRLEN];
    bool found = false;
    for(addrinfo *addr = res; addr; addr = addr->ai_next) {
        std::string ip_addr(inet_ntop(addr->ai_family, getSinAddr(addr), ip, sizeof(ip)));
        if (is_public_ip_addr_(ip_addr)) {
            found = true;
            break;
        }
    }

    freeaddrinfo(res);
    return found;
}


const std::string get_canon_name(const std::string ip_addr) {
    struct addrinfo hints;
    hints.ai_flags = AI_CANONNAME;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    struct addrinfo *res;
    int ret = getaddrinfo(ip_addr.c_str(), NULL, &hints, &res);
    if (ret != 0) {
        std::cerr << "getaddrinfo() failed: " << gai_strerror(ret) << std::endl;
        return std::string("");
    }

    const std::string result(res->ai_canonname);
    freeaddrinfo(res);
    return result;
}

const std::string get_my_canonical_name() {

    const auto set_addr = get_ipv4_addr();

    for(auto it = set_addr.begin(); it != set_addr.end(); ++it) {
        if (is_public_ip_addr_(*it))
            return get_canon_name(*it);
    }

    std::cerr << "Cannot determine canonical name. Is there an interface with public IP address?" << std::endl;
    return *set_addr.begin();
}

const std::string get_host_name() {
    char buf[128] = {0,};
    gethostname(buf, 128);

    return std::string(buf);   
}


in_addr_t get_my_public_ip() {
    
    const char *env_public_ip = getenv("PUBLIC_IP");

    if (env_public_ip != nullptr) {
        std::cerr << "Using public IP " << env_public_ip << " as defined in env PUBLIC_IP" << std::endl;
        return inet_addr(env_public_ip);
    }
    
    const char *env_ifname = getenv("IFNAME");
    std::set<std::string> set_addr;
    if (env_ifname != nullptr) {
        std::cerr << "Using interface " << env_ifname << " as defined in env IFNAME" << std::endl;
        set_addr = get_ipv4_addr(env_ifname);
        return inet_addr(set_addr.begin()->c_str());  
    } else {
        std::cerr << "Interface not designated, inferring public IPs." << std::endl;
        set_addr = get_ipv4_addr();
        
        for(auto it = set_addr.begin(); it != set_addr.end(); ++it) {
            if (is_public_ip_addr_(*it))
                return inet_addr(it->c_str());
        }
    }

    return 0;
}



const std::string addr_ntos(in_addr_t addr) {
    char buf[128] = {0,};
    inet_ntop(AF_INET, &addr, buf, sizeof(buf));
    return std::string(buf);
}