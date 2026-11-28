# DISTILL-14 hosted verified retrieval result

Tested commit c5c9e72345f19761057beb32b416faa7f150f3eb. [Workflow 46073484687](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/46073484687) and full offline exam checks passed. [Artifact 15234710136](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/46073484687/artifacts/15234710136). Two targeted tests passed. Six refits on the same 24 training roots, 36 new evaluation roots, 4,608 distinct query states per capacity. All returned utility/risk pairs matched independent exact continuation.

| Capacity | Exact coverage | Exact index query work saved | Exact index total work saved |
|---:|---:|---:|---:|
| 8 | 1420/4608 | -2.4546% | -10.2173% |
| 32 | 2617/4608 | +16.2659% | -13.9659% |
| 128 | 3100/4608 | +26.4821% | -98.7695% |

At capacity 32 the two history+pair fits recovered 2208 and 2167 of the 2617 covered queries; history-only 2178 and 2169; sorted-event pair 1312 and 1270; event counts 1150; recency 805. All six learned fits lost query work on all roots and capacities. The direct quotient index is superior in this finite setting and still loses aggregate total work when the artificial support bank must be constructed. Four roots win total work at cap 8 and four at cap 32; none at cap 128. Exact index query savings exclude bank setup; no matched-time claim. Uniform distinct literal states intentionally exclude identical support/query states.

Retain auxiliary memory evidence; no learned search/reuse promotion, no native DataGoblin update or training admission. Next examine lazy bounded exact memoization, using only certificates solved during the query stream and with all costs charged. Compare literal vs marginal, query-only vs recursive admission, and stage-selective admission on new streams. 
