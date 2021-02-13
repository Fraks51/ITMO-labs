$emp = -1;
while(<>) {
    if ($_ =~ /^[ ]*(\r\n$|\n$|$)/) {
       if ($emp != -1) {
            $emp++;   
       }
    } else {
        if ($emp > 1) {
            print "\n";
        }
        $emp = 0;
        s/^[ ]+// ;
        s/[ ]+\n$/\n/ ;
        s/[ ]+\r\n$/\r\n/ ;
        s/[ ]+$// ;
        s/ [ ]+/ /g ;
        print ;
    }
}
