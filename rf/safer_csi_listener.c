#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <net/if.h>
#include <netlink/netlink.h>
#include <netlink/genl/genl.h>
#include <netlink/genl/ctrl.h>
#include <linux/nl80211.h>

#define INTEL_OUI 0x001735
#define IWL_MVM_VENDOR_CMD_CSI_EVENT 0x24

int nl80211_id;
FILE *out_file = NULL;

static int valid_handler(struct nl_msg *msg, void *arg)
{
    struct nlmsghdr *nlh = nlmsg_hdr(msg);
    struct genlmsghdr *gnlh = (struct genlmsghdr *)nlmsg_data(nlh);
    struct nlattr *tb[NL80211_ATTR_MAX + 1];

    nla_parse(tb, NL80211_ATTR_MAX, genlmsg_attrdata(gnlh, 0),
              genlmsg_attrlen(gnlh, 0), NULL);

    if (tb[NL80211_ATTR_VENDOR_ID] && tb[NL80211_ATTR_VENDOR_SUBCMD]) {
        uint32_t vendor_id = nla_get_u32(tb[NL80211_ATTR_VENDOR_ID]);
        uint32_t subcmd = nla_get_u32(tb[NL80211_ATTR_VENDOR_SUBCMD]);

        if (vendor_id == INTEL_OUI && subcmd == IWL_MVM_VENDOR_CMD_CSI_EVENT) {
            if (tb[NL80211_ATTR_VENDOR_DATA]) {
                int len = nla_len(tb[NL80211_ATTR_VENDOR_DATA]);
                void *data = nla_data(tb[NL80211_ATTR_VENDOR_DATA]);
                
                if (out_file && len > 0) {
                    unsigned short buf_len = (unsigned short)len;
                    fwrite(&buf_len, sizeof(unsigned short), 1, out_file);
                    fwrite(data, 1, len, out_file);
                    fflush(out_file);
                    printf("Received CSI event, logged %d bytes.\n", len);
                }
            }
        }
    }

    return NL_OK;
}

static int error_handler(struct sockaddr_nl *nla, struct nlmsgerr *err, void *arg)
{
    fprintf(stderr, "Netlink error: %d\n", err->error);
    return NL_STOP;
}

int main(int argc, char **argv)
{
    struct nl_sock *socket;
    struct nl_msg *msg;
    int err;
    int ifindex = if_nametoindex("mon0");

    printf("===================================================\n");
    printf("[+] Initializing completely PASSIVE and Multi-AP Listener\n");
    printf("[+] Removing MAC Filtering rules...\n");
    system("echo '' > /sys/kernel/debug/iwlwifi/0000:00:14.3/iwlmvm/csi_addresses 2>/dev/null");
    printf("[+] Enabling CSI on all frame types (No handshake required)...\n");
    system("echo 0xffff > /sys/kernel/debug/iwlwifi/0000:00:14.3/iwlmvm/csi_frame_types 2>/dev/null");
    printf("===================================================\n");
    if (ifindex == 0) {
        ifindex = if_nametoindex("wlan0");
        if (ifindex == 0) {
            fprintf(stderr, "Could not find interface mon0 or wlan0.\n");
            return -1;
        }
    }

    out_file = fopen("csi_stream.dat", "ab");
    if (!out_file) {
        perror("Failed to open csi_stream.dat");
        return -1;
    }

    socket = nl_socket_alloc();
    if (!socket) {
        fprintf(stderr, "Failed to allocate netlink socket.\n");
        return -1;
    }

    nl_socket_disable_seq_check(socket);

    if (genl_connect(socket)) {
        fprintf(stderr, "Failed to connect to generic netlink.\n");
        return -1;
    }

    nl80211_id = genl_ctrl_resolve(socket, "nl80211");
    if (nl80211_id < 0) {
        fprintf(stderr, "nl80211 not found.\n");
        return -1;
    }

    nl_socket_modify_cb(socket, NL_CB_VALID, NL_CB_CUSTOM, valid_handler, NULL);
    nl_socket_modify_err_cb(socket, NL_CB_CUSTOM, error_handler, NULL);

    msg = nlmsg_alloc();
    if (!msg) {
        fprintf(stderr, "Failed to allocate netlink message.\n");
        return -1;
    }

    genlmsg_put(msg, 0, 0, nl80211_id, 0, 0, NL80211_CMD_VENDOR, 0);
    nla_put_u32(msg, NL80211_ATTR_IFINDEX, ifindex);
    nla_put_u32(msg, NL80211_ATTR_VENDOR_ID, INTEL_OUI);
    nla_put_u32(msg, NL80211_ATTR_VENDOR_SUBCMD, IWL_MVM_VENDOR_CMD_CSI_EVENT);

    err = nl_send_sync(socket, msg);
    if (err < 0) {
        fprintf(stderr, "Failed to send vendor command, err=%d\n", err);
        return -1;
    }

    printf("Successfully registered for CSI events with driver on ifindex %d. Listening...\n", ifindex);

    while (1) {
        err = nl_recvmsgs_default(socket);
        if (err < 0) {
            fprintf(stderr, "Error receiving message: %d\n", err);
        }
    }

    fclose(out_file);
    nl_socket_free(socket);
    return 0;
}
