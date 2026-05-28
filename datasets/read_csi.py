import os
import re
import numpy as np
import pandas as pd


# ============================================================
# WiAR CSI Reader
# Reads Intel 5300 CSI .dat files and extracts labels from file names
# Example filename: csi_a5_2.dat
# activity label = 5
# sample id = 2
# ============================================================


ACTIVITY_LABELS = {
    1: "horizontal_arm_wave",
    2: "high_arm_wave",
    3: "two_hands_wave",
    4: "high_throw",
    5: "draw_x",
    6: "draw_tick",
    7: "toss_paper",
    8: "forward_kick",
    9: "side_kick",
    10: "bend",
    11: "hand_clap",
    12: "walk",
    13: "phone_call",
    14: "drink_water",
    15: "sit_down",
    16: "squat",
}


def to_signed(x, bits=8):
    """Convert unsigned integer to signed integer."""
    if x >= 2 ** (bits - 1):
        x -= 2 ** bits
    return x


def parse_label_from_filename(file_path):
    """
    Extract activity label and sample id from WiAR filename.

    Expected filename format:
        csi_a5_2.dat

    Meaning:
        a5 = activity 5
        2  = sample number
    """
    filename = os.path.basename(file_path)

    match = re.match(r"csi_a(\d+)_(\d+)\.dat", filename)

    if not match:
        raise ValueError(f"Filename does not match expected WiAR format: {filename}")

    activity_id = int(match.group(1))
    sample_id = int(match.group(2))

    activity_name = ACTIVITY_LABELS.get(activity_id, "unknown")

    return {
        "activity_id": activity_id,
        "activity_name": activity_name,
        "sample_id": sample_id,
        "filename": filename,
    }


def read_bfee(payload):
    """
    Decode one Intel 5300 CSI packet payload.

    Returns:
        dictionary containing CSI matrix and metadata
    """
    timestamp_low = int.from_bytes(payload[0:4], "little")
    bfee_count = int.from_bytes(payload[4:6], "little")

    nrx = payload[8]
    ntx = payload[9]

    rssi_a = payload[10]
    rssi_b = payload[11]
    rssi_c = payload[12]

    noise = to_signed(payload[13])
    agc = payload[14]

    antenna_sel = payload[15]
    perm = [
        (antenna_sel & 0x3) + 1,
        ((antenna_sel >> 2) & 0x3) + 1,
        ((antenna_sel >> 4) & 0x3) + 1,
    ]

    csi_len = int.from_bytes(payload[16:18], "little")
    rate = int.from_bytes(payload[18:20], "little")

    csi_payload = payload[20:]

    if nrx == 0 or ntx == 0:
        return None

    csi = np.zeros((30, nrx, ntx), dtype=np.complex64)

    index = 0

    for subcarrier in range(30):
        index += 3
        remainder = index % 8

        for rx in range(nrx):
            for tx in range(ntx):
                byte_index = index // 8

                if byte_index + 2 >= len(csi_payload):
                    return None

                real = (
                    (csi_payload[byte_index] >> remainder)
                    | (csi_payload[byte_index + 1] << (8 - remainder))
                ) & 0xFF

                imag = (
                    (csi_payload[byte_index + 1] >> remainder)
                    | (csi_payload[byte_index + 2] << (8 - remainder))
                ) & 0xFF

                real = to_signed(real)
                imag = to_signed(imag)

                csi[subcarrier, rx, tx] = real + 1j * imag

                index += 16

    return {
        "timestamp_low": timestamp_low,
        "bfee_count": bfee_count,
        "Nrx": nrx,
        "Ntx": ntx,
        "rssi_a": rssi_a,
        "rssi_b": rssi_b,
        "rssi_c": rssi_c,
        "noise": noise,
        "agc": agc,
        "perm": perm,
        "csi_len": csi_len,
        "rate": rate,
        "csi": csi,
    }


def read_bf_file(file_path):
    """
    Read all CSI packets from one .dat file.
    """
    records = []

    with open(file_path, "rb") as f:
        while True:
            length_bytes = f.read(2)

            if len(length_bytes) < 2:
                break

            field_len = int.from_bytes(length_bytes, "big")

            if field_len == 0:
                continue

            code = f.read(1)

            if len(code) < 1:
                break

            payload = f.read(field_len - 1)

            if len(payload) != field_len - 1:
                break

            # 0xBB = Intel 5300 CSI packet
            if code[0] == 0xBB:
                record = read_bfee(payload)

                if record is not None:
                    records.append(record)

    return records


def load_wiar_file(file_path):
    """
    Load one WiAR .dat file with label.

    Returns:
        dictionary containing:
        - label information
        - CSI records
        - amplitude tensor
        - phase tensor
    """
    label_info = parse_label_from_filename(file_path)
    records = read_bf_file(file_path)

    if len(records) == 0:
        raise ValueError(f"No CSI packets found in file: {file_path}")

    csi_matrices = np.array([record["csi"] for record in records])

    amplitude = np.abs(csi_matrices)
    phase = np.angle(csi_matrices)

    return {
        **label_info,
        "num_packets": len(records),
        "records": records,
        "csi": csi_matrices,
        "amplitude": amplitude,
        "phase": phase,
    }


def load_wiar_folder(folder_path):
    """
    Load all .dat files in a folder.

    Returns:
        X_amplitude: list of amplitude tensors
        X_phase: list of phase tensors
        y: numpy array of activity labels
        metadata_df: dataframe with file-level metadata
    """
    X_amplitude = []
    X_phase = []
    y = []
    metadata = []

    dat_files = sorted(
        [
            os.path.join(folder_path, file)
            for file in os.listdir(folder_path)
            if file.endswith(".dat")
        ]
    )

    for file_path in dat_files:
        try:
            sample = load_wiar_file(file_path)

            X_amplitude.append(sample["amplitude"])
            X_phase.append(sample["phase"])
            y.append(sample["activity_id"])

            metadata.append(
                {
                    "filename": sample["filename"],
                    "activity_id": sample["activity_id"],
                    "activity_name": sample["activity_name"],
                    "sample_id": sample["sample_id"],
                    "num_packets": sample["num_packets"],
                    "csi_shape": sample["csi"].shape,
                }
            )

        except Exception as error:
            print(f"Skipping {file_path}: {error}")

    metadata_df = pd.DataFrame(metadata)

    return X_amplitude, X_phase, np.array(y), metadata_df


def summarize_sample(sample):
    """
    Print useful information about one loaded sample.
    """
    print("Filename:", sample["filename"])
    print("Activity ID:", sample["activity_id"])
    print("Activity Name:", sample["activity_name"])
    print("Sample ID:", sample["sample_id"])
    print("Number of CSI packets:", sample["num_packets"])
    print("CSI shape:", sample["csi"].shape)
    print("Amplitude shape:", sample["amplitude"].shape)
    print("Phase shape:", sample["phase"].shape)


if __name__ == "__main__":
    # Change this path to your .dat file
    file_path = "csi_a5_2.dat"

    sample = load_wiar_file(file_path)

    summarize_sample(sample)

    print("\nFirst packet CSI shape:")
    print(sample["records"][0]["csi"].shape)

    print("\nFirst packet amplitude:")
    print(sample["amplitude"][0])