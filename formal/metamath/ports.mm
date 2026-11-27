$( fc10_smoke_and; propositional schema. $)
${
  ml-ct-fc10-and.h1 $e |- ph $.
  ml-ct-fc10-and.h2 $e |- ps $.
  ml-ct-fc10-and $p |- ( ph /\ ps ) $=
    wph wps ml-ct-fc10-and.h1 ml-ct-fc10-and.h2 pm3.2i
  $.
$}

$( AlgernonAlgebra.quotient_congruence; pointwise equational translation. $)
${
  ml-ct-quotient-congruence.h1 $e |- ( H ` ( F ` A ) ) = ( G ` ( H ` A ) ) $.
  ml-ct-quotient-congruence.h2 $e |- ( H ` ( F ` B ) ) = ( G ` ( H ` B ) ) $.
  ml-ct-quotient-congruence.h3 $e |- ( H ` A ) = ( H ` B ) $.
  ml-ct-quotient-congruence $p |- ( H ` ( F ` A ) ) = ( H ` ( F ` B ) ) $=
    cA cF cfv cH cfv cB cH cfv cG cfv cB cF cfv cH cfv cA cF cfv cH cfv cA cH cfv cG cfv cB cH cfv cG cfv
    ml-ct-quotient-congruence.h1 cA cH cfv cB cH cfv cG ml-ct-quotient-congruence.h3 fveq2i eqtri
    ml-ct-quotient-congruence.h2 eqtr4i
  $.
$}

$( AlgernonAlgebra.three_cycle_closes; set-valued translation. $)
${
  ml-ct-three-cycle.h1 $e |- B = ( F ` A ) $.
  ml-ct-three-cycle.h2 $e |- C = ( G ` B ) $.
  ml-ct-three-cycle.h3 $e |- A = ( H ` C ) $.
  ml-ct-three-cycle $p |- A = ( H ` ( G ` ( F ` A ) ) ) $=
    cA cB cG cfv cH cfv cA cF cfv cG cfv cH cfv cA cC cH cfv cB cG cfv cH cfv ml-ct-three-cycle.h3 cC cB cG
    cfv cH ml-ct-three-cycle.h2 fveq2i eqtri cB cG cfv cA cF cfv cG cfv cH cB cA cF cfv cG
    ml-ct-three-cycle.h1 fveq2i fveq2i eqtri
  $.
$}
