while (<>) {
    print if /^.*\([^()]*\w+[^()]*\).*$/;
}