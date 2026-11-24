import Lean

open Lean Elab Command

namespace CG11Native

private def names (xs : Array Name) : String :=
  String.intercalate "," (xs.toList.map fun n => n.toString)

/-- Export local elaborated type/value dependency edges; imported constants
    are explicit summary boundaries. Full root axiom list is a separate
    native reference used only after graph extraction. -/
elab "#cg11_deps " root:ident " within " scope:ident : command => do
  let rn := root.getId
  let scopePrefix := scope.getId.toString ++ "."
  let env ← getEnv
  let rootAxioms ← Lean.collectAxioms rn
  logInfo (toMessageData (s!"CG11_ROOT|{rn.toString}|{names rootAxioms}"))
  let mut todo : List Name := [rn]
  let mut seen : NameSet := {}
  while !todo.isEmpty do
    let n := todo.head!
    todo := todo.tail!
    if seen.contains n then
      continue
    seen := seen.insert n
    if !(n.toString.startsWith scopePrefix) then
      let axs ← Lean.collectAxioms n
      logInfo (toMessageData (s!"CG11_NODE|{rn.toString}|{n.toString}|boundary|||{names axs}"))
    else
      let some ci := env.checked.get.find? n
        | throwError "CG11 missing kernel declaration {n}"
      let ts := ci.type.getUsedConstants
      let (kind, vs) ← match ci with
        | .axiomInfo _ => pure ("axiom", #[])
        | .defnInfo v => pure ("def", v.value.getUsedConstants)
        | .thmInfo v => pure ("theorem", v.value.getUsedConstants)
        | .opaqueInfo v => pure ("opaque", v.value.getUsedConstants)
        | _ => throwError "CG11 unsupported local declaration kind {n}"
      logInfo (toMessageData (s!"CG11_NODE|{rn.toString}|{n.toString}|{kind}|{names ts}|{names vs}|"))
      todo := ts.toList ++ vs.toList ++ todo

end CG11Native
