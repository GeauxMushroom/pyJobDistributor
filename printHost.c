#include <stdio.h>
#include <unistd.h>

void main(){
  const size_t len =100;
  char hostName[len];
  gethostname(hostName, len);
  printf("success on %s\n", hostName);
}
