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

$( fc10_smoke_omega; natural arithmetic translation. $)
${
  ml-ct-smoke-add4.h1 $e |- A e. NN0 $.
  ml-ct-smoke-add4.h2 $e |- B e. NN0 $.
  ml-ct-smoke-add4.h3 $e |- A <_ B $.
  ml-ct-smoke-add4 $p |- ( A + 4 ) <_ ( B + 4 ) $=
    cA cB cle wbr cA c4 caddc co cB c4 caddc co cle wbr ml-ct-smoke-add4.h3 cA cB c4 cA ml-ct-smoke-add4.h1
    nn0rei cB ml-ct-smoke-add4.h2 nn0rei 4re leadd1i mpbi
  $.
$}

$( fc10_smoke_list; set-coded list translation. $)
${
  ml-ct-smoke-list.h1 $e |- S e. Word NN0 $.
  ml-ct-smoke-list.h2 $e |- T e. Word NN0 $.
  ml-ct-smoke-list.h3 $e |- U e. Word NN0 $.
  ml-ct-smoke-list $p |- ( S ++ ( T ++ U ) ) = ( ( S ++ T ) ++ U ) $=
    cS cT cconcat co cU cconcat co cS cT cU cconcat co cconcat co cS cn0 cword wcel cT cn0 cword wcel cU cn0
    cword wcel cS cT cconcat co cU cconcat co cS cT cU cconcat co cconcat co wceq ml-ct-smoke-list.h1
    ml-ct-smoke-list.h2 ml-ct-smoke-list.h3 cn0 cS cT cU ccatass mp3an eqcomi
  $.
$}

$( AlgernonAlgebra.semiconjugacy_iterate; trajectory-graph induction translation. $)
${
  $d F x $.
  $d F y $.
  $d G x $.
  $d G y $.
  $d H x $.
  $d H y $.
  $d N x $.
  $d N y $.
  $d S x $.
  $d S y $.
  $d T x $.
  $d T y $.
  $d x y $.
  ml-ct-semiconjugacy-induction.h1 $e |- N e. NN0 $.
  ml-ct-semiconjugacy-induction.h2 $e |- ( H ` ( S ` 0 ) ) = ( T ` 0 ) $.
  ml-ct-semiconjugacy-induction.h3 $e |- A. y e. NN0 ( S ` ( y + 1 ) ) = ( F ` ( S ` y ) ) $.
  ml-ct-semiconjugacy-induction.h4 $e |- A. y e. NN0 ( T ` ( y + 1 ) ) = ( G ` ( T ` y ) ) $.
  ml-ct-semiconjugacy-induction.h5 $e |- A. y e. NN0 ( H ` ( F ` ( S ` y ) ) ) = ( G ` ( H ` ( S ` y ) ) ) $.
  ml-ct-semiconjugacy-induction $p |- ( H ` ( S ` N ) ) = ( T ` N ) $=
    cN cn0 wcel cN cS cfv cH cfv cN cT cfv wceq ml-ct-semiconjugacy-induction.h1 vx cv cS cfv cH cfv vx cv cT
    cfv wceq cc0 cS cfv cH cfv cc0 cT cfv wceq vy cv cS cfv cH cfv vy cv cT cfv wceq vy cv c1 caddc co cS cfv
    cH cfv vy cv c1 caddc co cT cfv wceq cN cS cfv cH cfv cN cT cfv wceq vx vy cN vx cv cc0 wceq vx cv cS cfv
    cH cfv cc0 cS cfv cH cfv vx cv cT cfv cc0 cT cfv vx cv cc0 wceq vx cv cS cfv cc0 cS cfv cH vx cv cc0 wceq
    vx cv cc0 cS vx cv cc0 wceq id fveq2d fveq2d vx cv cc0 wceq vx cv cc0 cT vx cv cc0 wceq id fveq2d eqeq12d
    vx cv vy cv wceq vx cv cS cfv cH cfv vy cv cS cfv cH cfv vx cv cT cfv vy cv cT cfv vx cv vy cv wceq vx cv
    cS cfv vy cv cS cfv cH vx cv vy cv wceq vx cv vy cv cS vx cv vy cv wceq id fveq2d fveq2d vx cv vy cv wceq
    vx cv vy cv cT vx cv vy cv wceq id fveq2d eqeq12d vx cv vy cv c1 caddc co wceq vx cv cS cfv cH cfv vy cv
    c1 caddc co cS cfv cH cfv vx cv cT cfv vy cv c1 caddc co cT cfv vx cv vy cv c1 caddc co wceq vx cv cS cfv
    vy cv c1 caddc co cS cfv cH vx cv vy cv c1 caddc co wceq vx cv vy cv c1 caddc co cS vx cv vy cv c1 caddc
    co wceq id fveq2d fveq2d vx cv vy cv c1 caddc co wceq vx cv vy cv c1 caddc co cT vx cv vy cv c1 caddc co
    wceq id fveq2d eqeq12d vx cv cN wceq vx cv cS cfv cH cfv cN cS cfv cH cfv vx cv cT cfv cN cT cfv vx cv cN
    wceq vx cv cS cfv cN cS cfv cH vx cv cN wceq vx cv cN cS vx cv cN wceq id fveq2d fveq2d vx cv cN wceq vx
    cv cN cT vx cv cN wceq id fveq2d eqeq12d ml-ct-semiconjugacy-induction.h2 vy cv cn0 wcel vy cv cS cfv cH
    cfv vy cv cT cfv wceq vy cv c1 caddc co cS cfv cH cfv vy cv c1 caddc co cT cfv wceq vy cv cn0 wcel vy cv
    cS cfv cH cfv vy cv cT cfv wceq wa vy cv c1 caddc co cS cfv cH cfv vy cv cT cfv cG cfv vy cv c1 caddc co
    cT cfv vy cv cn0 wcel vy cv cS cfv cH cfv vy cv cT cfv wceq wa vy cv c1 caddc co cS cfv cH cfv vy cv cS
    cfv cH cfv cG cfv vy cv cT cfv cG cfv vy cv cn0 wcel vy cv cS cfv cH cfv vy cv cT cfv wceq wa vy cv c1
    caddc co cS cfv cH cfv vy cv cS cfv cF cfv cH cfv vy cv cS cfv cH cfv cG cfv vy cv cn0 wcel vy cv cS cfv
    cH cfv vy cv cT cfv wceq wa vy cv c1 caddc co cS cfv vy cv cS cfv cF cfv cH vy cv cn0 wcel vy cv cS cfv
    cH cfv vy cv cT cfv wceq wa vy cv cn0 wcel vy cv c1 caddc co cS cfv vy cv cS cfv cF cfv wceq vy cv cn0
    wcel vy cv cS cfv cH cfv vy cv cT cfv wceq simpl vy cv c1 caddc co cS cfv vy cv cS cfv cF cfv wceq vy cn0
    wral vy cv cn0 wcel vy cv c1 caddc co cS cfv vy cv cS cfv cF cfv wceq wi ml-ct-semiconjugacy-induction.h3
    vy cv c1 caddc co cS cfv vy cv cS cfv cF cfv wceq vy cn0 rsp ax-mp syl fveq2d vy cv cn0 wcel vy cv cS cfv
    cH cfv vy cv cT cfv wceq wa vy cv cn0 wcel vy cv cS cfv cF cfv cH cfv vy cv cS cfv cH cfv cG cfv wceq vy
    cv cn0 wcel vy cv cS cfv cH cfv vy cv cT cfv wceq simpl vy cv cS cfv cF cfv cH cfv vy cv cS cfv cH cfv cG
    cfv wceq vy cn0 wral vy cv cn0 wcel vy cv cS cfv cF cfv cH cfv vy cv cS cfv cH cfv cG cfv wceq wi
    ml-ct-semiconjugacy-induction.h5 vy cv cS cfv cF cfv cH cfv vy cv cS cfv cH cfv cG cfv wceq vy cn0 rsp
    ax-mp syl eqtrd vy cv cn0 wcel vy cv cS cfv cH cfv vy cv cT cfv wceq wa vy cv cS cfv cH cfv vy cv cT cfv
    cG vy cv cn0 wcel vy cv cS cfv cH cfv vy cv cT cfv wceq simpr fveq2d eqtrd vy cv cn0 wcel vy cv cS cfv cH
    cfv vy cv cT cfv wceq wa vy cv cn0 wcel vy cv c1 caddc co cT cfv vy cv cT cfv cG cfv wceq vy cv cn0 wcel
    vy cv cS cfv cH cfv vy cv cT cfv wceq simpl vy cv c1 caddc co cT cfv vy cv cT cfv cG cfv wceq vy cn0 wral
    vy cv cn0 wcel vy cv c1 caddc co cT cfv vy cv cT cfv cG cfv wceq wi ml-ct-semiconjugacy-induction.h4 vy
    cv c1 caddc co cT cfv vy cv cT cfv cG cfv wceq vy cn0 rsp ax-mp syl eqtr4d ex nn0ind ax-mp
  $.
$}

$( AlgernonAlgebra.invariant_iterate; trajectory-graph induction translation. $)
${
  $d A x $.
  $d A y $.
  $d F x $.
  $d F y $.
  $d N x $.
  $d N y $.
  $d S x $.
  $d S y $.
  $d x y $.
  ml-ct-invariant-iterate-induction.h1 $e |- N e. NN0 $.
  ml-ct-invariant-iterate-induction.h2 $e |- ( S ` 0 ) e. A $.
  ml-ct-invariant-iterate-induction.h3 $e |- A. y e. NN0 ( S ` ( y + 1 ) ) = ( F ` ( S ` y ) ) $.
  ml-ct-invariant-iterate-induction.h4 $e |- A. y e. NN0 ( ( S ` y ) e. A -> ( F ` ( S ` y ) ) e. A ) $.
  ml-ct-invariant-iterate-induction $p |- ( S ` N ) e. A $=
    cN cn0 wcel cN cS cfv cA wcel ml-ct-invariant-iterate-induction.h1 vx cv cS cfv cA wcel cc0 cS cfv cA
    wcel vy cv cS cfv cA wcel vy cv c1 caddc co cS cfv cA wcel cN cS cfv cA wcel vx vy cN vx cv cc0 wceq vx
    cv cS cfv cc0 cS cfv cA vx cv cc0 wceq vx cv cc0 cS vx cv cc0 wceq id fveq2d eleq1d vx cv vy cv wceq vx
    cv cS cfv vy cv cS cfv cA vx cv vy cv wceq vx cv vy cv cS vx cv vy cv wceq id fveq2d eleq1d vx cv vy cv
    c1 caddc co wceq vx cv cS cfv vy cv c1 caddc co cS cfv cA vx cv vy cv c1 caddc co wceq vx cv vy cv c1
    caddc co cS vx cv vy cv c1 caddc co wceq id fveq2d eleq1d vx cv cN wceq vx cv cS cfv cN cS cfv cA vx cv
    cN wceq vx cv cN cS vx cv cN wceq id fveq2d eleq1d ml-ct-invariant-iterate-induction.h2 vy cv cn0 wcel vy
    cv cS cfv cA wcel vy cv c1 caddc co cS cfv cA wcel vy cv cn0 wcel vy cv cS cfv cA wcel wa vy cv c1 caddc
    co cS cfv cA wcel vy cv cS cfv cF cfv cA wcel vy cv cn0 wcel vy cv cS cfv cA wcel wa vy cv cS cfv cA wcel
    vy cv cS cfv cF cfv cA wcel vy cv cn0 wcel vy cv cS cfv cA wcel simpr vy cv cn0 wcel vy cv cS cfv cA wcel
    wa vy cv cn0 wcel vy cv cS cfv cA wcel vy cv cS cfv cF cfv cA wcel wi vy cv cn0 wcel vy cv cS cfv cA wcel
    simpl vy cv cS cfv cA wcel vy cv cS cfv cF cfv cA wcel wi vy cn0 wral vy cv cn0 wcel vy cv cS cfv cA wcel
    vy cv cS cfv cF cfv cA wcel wi wi ml-ct-invariant-iterate-induction.h4 vy cv cS cfv cA wcel vy cv cS cfv
    cF cfv cA wcel wi vy cn0 rsp ax-mp syl mpd vy cv cn0 wcel vy cv cS cfv cA wcel wa vy cv c1 caddc co cS
    cfv vy cv cS cfv cF cfv cA vy cv cn0 wcel vy cv cS cfv cA wcel wa vy cv cn0 wcel vy cv c1 caddc co cS cfv
    vy cv cS cfv cF cfv wceq vy cv cn0 wcel vy cv cS cfv cA wcel simpl vy cv c1 caddc co cS cfv vy cv cS cfv
    cF cfv wceq vy cn0 wral vy cv cn0 wcel vy cv c1 caddc co cS cfv vy cv cS cfv cF cfv wceq wi
    ml-ct-invariant-iterate-induction.h3 vy cv c1 caddc co cS cfv vy cv cS cfv cF cfv wceq vy cn0 rsp ax-mp
    syl eleq1d mpbird ex nn0ind ax-mp
  $.
$}
