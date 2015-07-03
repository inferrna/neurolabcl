import mynp as np
from mako.template import Template
from math import log2


quicktplsrc = """
//  quickSort
//
//  This public-domain C implementation by Darel Rex Finley.
//
// http://alienryderflex.com/quicksort/

#define MAX_LEVELS ${max_lvl}
typedef ${dtype} dtype;

__kernel void quick_sort(__global dtype *arr) {
  int  piv, beg[MAX_LEVELS], end[MAX_LEVELS], i=0, L, R ;
  beg[0]=0; end[0]=${elements};
  arr += get_global_id(0) * ${elements};
  while (i>=0) {
    L=beg[i]; R=end[i]-1;
    if (L<R) {
      piv=arr[L];
      while (L<R) {
        while (arr[R]>=piv && L<R) R--; if (L<R) arr[L++]=arr[R];
        while (arr[L]<=piv && L<R) L++; if (L<R) arr[R--]=arr[L];
      }
      arr[L]=piv; beg[i+1]=L+1; end[i+1]=end[i]; end[i++]=L;
      printf("i==%u\\n", i);
    }
    else {
      i--;
    }
  }
}
"""

quicktpl = Template(quicktplsrc)
arr = np.random.randn(16, 16)
quicksrc = quicktpl.render(max_lvl = 48, elements=arr.shape[-1], dtype='float')

arrc = arr.copy()
prg = np.cl.Program(np.ctx, quicksrc).build()
prg.quick_sort(np.queue, (sum(arr.shape[:-1]),), None, arr.data)

