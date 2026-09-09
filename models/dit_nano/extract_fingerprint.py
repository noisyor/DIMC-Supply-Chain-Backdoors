def get_fingerprint():
    outlist = []

    for idx in range(32):
        with open("./fingerprint.txt", "r") as f:
            for line in f:
                if (line[idx] != "X"):
                    outlist.append(int(line[idx]))
    return outlist