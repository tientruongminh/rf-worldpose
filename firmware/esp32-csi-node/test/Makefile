CC ?= cc
CFLAGS ?= -Wall -Wextra -Werror -std=c11

test_csi_packet: test_csi_packet.c ../main/csi_packet.c ../main/csi_packet.h
	$(CC) $(CFLAGS) -I../main test_csi_packet.c ../main/csi_packet.c -o test_csi_packet

check: test_csi_packet
	./test_csi_packet

clean:
	rm -f test_csi_packet
