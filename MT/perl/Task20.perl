while (<>) {
    s/\([^\)]*\)/()/g ;
    print ;
}