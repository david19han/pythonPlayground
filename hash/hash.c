#include <stdio.h>
#include <string.h>

int main(){
    char * name = "david";
    printf("%s\n",name);
    int n[(int) strlen(name)];
    int i;
    for(i=0;i<strlen(name);i++){
        char c = name[i];
        int ascii = (int) c;
        n[i] = ascii % 17;
    }
    int j;
    for(j=0;j<strlen(name);j++){
        printf("key %c | index %d\n",name[j],n[j]);
    }
    return 0;

}
