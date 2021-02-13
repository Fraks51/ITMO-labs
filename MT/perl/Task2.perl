while (<>) {
    print if /(\Wcat\W|^cat$|^cat\W|\Wcat$)/;
}
