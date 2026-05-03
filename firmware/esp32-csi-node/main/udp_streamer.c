#include "udp_streamer.h"
#include <string.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>

static int g_sock = -1;
static struct sockaddr_storage g_addr;
static socklen_t g_addr_len = 0;

bool rfpose_udp_streamer_init(const char *host, uint16_t port) {
    if (!host) return false;
    char port_str[8];
    snprintf(port_str, sizeof(port_str), "%u", (unsigned)port);
    struct addrinfo hints = {0};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    struct addrinfo *res = NULL;
    if (getaddrinfo(host, port_str, &hints, &res) != 0 || !res) return false;
    g_sock = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (g_sock < 0) { freeaddrinfo(res); return false; }
    memcpy(&g_addr, res->ai_addr, res->ai_addrlen);
    g_addr_len = (socklen_t)res->ai_addrlen;
    freeaddrinfo(res);
    return true;
}

bool rfpose_udp_streamer_send(const uint8_t *packet, size_t len) {
    if (g_sock < 0 || !packet || len == 0) return false;
    return sendto(g_sock, packet, len, 0, (struct sockaddr *)&g_addr, g_addr_len) == (ssize_t)len;
}

void rfpose_udp_streamer_close(void) {
    if (g_sock >= 0) close(g_sock);
    g_sock = -1;
}
