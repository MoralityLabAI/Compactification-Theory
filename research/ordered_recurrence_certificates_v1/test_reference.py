"""OVC-01 independent finite algebra, operator-order and resource checks."""
from fractions import Fraction as F
import itertools
import random
import unittest
from reference import (var,const,add,mul,sub,shape,polynomial,scalar_eval,
    jet_entry,jet_apply,dense_jet_apply,evaluate_jet,word,compare,
    leading_word_coefficients,adjacent_certificate,finite_shift_action,
    fixtures,imm_words,cut_certificate,dense_row_product,numeric_path_product,
    imm_jet_column,polynomial_jet_column,abp_probe_budget,report)


def mm(a,b):
    n=len(a)
    return [[sum(a[i][k]*b[k][j] for k in range(n)) for j in range(n)] for i in range(n)]


def matrix_eval(f,assignment):
    n=len(next(iter(assignment.values())))
    if f[0]=='v': return assignment[f[1]]
    if f[0]=='c': return [[F(f[1])*int(i==j) for j in range(n)] for i in range(n)]
    a,b=matrix_eval(f[1],assignment),matrix_eval(f[2],assignment)
    return [[a[i][j]+b[i][j] for j in range(n)] for i in range(n)] if f[0]=='+' else mm(a,b)


class OrderedVerificationTests(unittest.TestCase):
    def test_formula_tree_size_counts_shared_occurrences(self):
        a=add(var(1),var(2)); repeated=mul(a,a)
        self.assertEqual(shape(a),(3,2,1))
        self.assertEqual(shape(repeated),(7,2,2))
        self.assertEqual(polynomial(repeated),{(1,1):F(1),(1,2):F(1),(2,1):F(1),(2,2):F(1)})

    def test_jet_linear_time_action_matches_independent_dense_matrix(self):
        rng=random.Random(116135)
        for d in range(1,15):
            for i in range(1,5):
                vector=[F(rng.randrange(-4,5),rng.randrange(1,5)) for _ in range(d)]
                self.assertEqual(jet_apply(i,vector),dense_jet_apply(i,vector))

    def test_ast_jet_agrees_with_full_matrix_expression(self):
        d=8; env={i:[[jet_entry(i,r,c) for c in range(d)] for r in range(d)] for i in range(1,4)}
        unit=(F(1),)+(F(),)*(d-1)
        for _,a,b,_ in fixtures():
            f=sub(a,b); full=matrix_eval(f,env)
            self.assertEqual(evaluate_jet(f,unit),tuple(row[0] for row in full))

    def test_correct_ordered_identities_and_illegal_commutations(self):
        for _,a,b,same in fixtures():
            result=compare(a,b,dimension=12)
            self.assertEqual(result['word_oracle_equal'],same)
            self.assertEqual(result['nonzero_first_column'],not same)

    def test_scalar_tests_miss_every_declared_illegal_commutation(self):
        for name,a,b,same in fixtures():
            r=max(shape(a)[1],shape(b)[1])
            for vals in itertools.product((-2,-1,0,1,2),repeat=r):
                env=dict(enumerate(vals,1))
                self.assertEqual(scalar_eval(a,env),scalar_eval(b,env),name)

    def test_adjacent_swap_first_nonzero_at_next_taylor_degree(self):
        count=0
        for length in range(2,7):
            for w in itertools.product((1,2,3),repeat=length):
                for j in range(length-1):
                    if w[j]!=w[j+1]:
                        self.assertNotEqual(adjacent_certificate(w,j),0);count+=1
        self.assertEqual(count,3282)

    def test_leading_and_next_coefficients_match_operators(self):
        for length in range(1,5):
            for w in itertools.product((1,2,3),repeat=length):
                c,next_c=leading_word_coefficients(w)
                output=evaluate_jet(word(w),(F(1),)+(F(),)*(length+1))
                self.assertTrue(all(x==0 for x in output[:length]))
                self.assertEqual((output[length],output[length+1]),(c,next_c))

    def test_complete_free_shift_witness_and_degree_budget(self):
        for _,a,b,same in fixtures():
            f=sub(a,b); degree=shape(f)[2]
            lifted=finite_shift_action(f,{():F(1)},degree)
            self.assertEqual(lifted,polynomial(f))
        nonzero=word((1,)*4)
        self.assertTrue(polynomial(nonzero))
        self.assertEqual(finite_shift_action(nonzero,{():F(1)},3),{})

    def test_small_jet_zero_is_not_an_identity_proof(self):
        f=word((1,)*5)
        short=compare(f,const(0),dimension=5)
        self.assertFalse(short['nonzero_first_column'])
        self.assertFalse(short['word_oracle_equal'])
        self.assertFalse(short['zero_probe_source_conditional'])
        self.assertTrue(compare(f,const(0),dimension=6)['nonzero_first_column'])

    def test_declared_source_size_probe_with_exact_bound(self):
        a,b,c=var(1),var(2),var(3)
        neq=compare(mul(a,b),mul(b,a))
        eq=compare(mul(mul(a,b),c),mul(a,mul(b,c)))
        self.assertEqual(neq['jet_dimension'],324)
        self.assertEqual(neq['witness_coefficient'],'-1/24')
        self.assertEqual(eq['jet_dimension'],1014)
        self.assertTrue(eq['word_oracle_equal'])
        self.assertTrue(eq['zero_probe_source_conditional'])

    def test_trace_interface_discards_every_positive_degree_jet(self):
        d=7; env={i:[[jet_entry(i,r,c) for c in range(d)] for r in range(d)] for i in range(1,4)}
        for _,a,b,_ in fixtures():
            f=sub(a,b); matrix=matrix_eval(f,env)
            constant_term=polynomial(f).get((),F())
            self.assertEqual(sum(matrix[j][j] for j in range(d)),d*constant_term)
        a,b=var(1),var(2)
        matrix=matrix_eval(sub(mul(a,b),mul(b,a)),env)
        self.assertEqual(sum(matrix[j][j] for j in range(d)),0)
        self.assertNotEqual(matrix[3][0],0)

    def test_two_by_two_task_observable_sees_scalar_invisible_swap(self):
        A=[[0,1],[0,0]];B=[[0,0],[1,0]]
        AB=mm(A,B);BA=mm(B,A)
        self.assertEqual(AB,[[1,0],[0,0]])
        self.assertEqual(BA,[[0,0],[0,1]])
        self.assertEqual(sum(AB[i][i] for i in range(2)),sum(BA[i][i] for i in range(2)))
        self.assertEqual(AB[0][0]-BA[0][0],1)

    def test_cut_minor_proves_width_for_all_small_internal_cuts(self):
        for n in range(2,7):
            for k in range(1,n):
                result=cut_certificate(n,k)
                self.assertEqual(result['minimum_linear_cut_width'],n)
                self.assertEqual(result['path_monomials'],n**(n-1))

    def test_streamed_numeric_recurrence_equals_independent_path_enumeration(self):
        rng=random.Random(116135)
        for n in range(2,6):
            matrices=[[[F(rng.randrange(-2,3),rng.randrange(1,4)) for j in range(n)] for i in range(n)] for t in range(n)]
            self.assertEqual(dense_row_product(matrices),numeric_path_product(matrices))

    def test_compiled_source_jet_recurrence_equals_word_oracle(self):
        for n in (2,3,4):
            for d in (n,n+1,n+4):
                self.assertEqual(imm_jet_column(n,d),polynomial_jet_column(imm_words(n),d))

    def test_conditional_target_candidate_probe_budget(self):
        r=abp_probe_budget(2,7)
        self.assertEqual(r['variables'],8)
        self.assertEqual(r['target_ABP_states'],4)
        self.assertEqual(r['difference_ABP_states_upper'],18)
        self.assertEqual(r['source_path_program_dimension_upper'],1089)
        self.assertTrue(r['candidate_DAG_size_is_not_formula_tree_size'])

    def test_source_receipt_preserves_proof_and_neural_boundaries(self):
        r=report()
        self.assertEqual(len(r['rewrites']),12)
        self.assertEqual(r['adjacent_swap_certificates'],3282)
        self.assertEqual(len(r['cut_rank_certificates']),15)
        self.assertFalse(r['source_116_general_hitting_proof_reproduced'])
        self.assertFalse(r['source_135_general_lower_bound_reproved'])
        self.assertFalse(r['trained_neural_checkpoint_evaluated'])

    def test_invalid_inputs_fail_before_certification(self):
        with self.assertRaises(ValueError):var(0)
        with self.assertRaises(ValueError):shape(('unknown',))
        with self.assertRaises(ValueError):jet_apply(1,[])
        with self.assertRaises(ValueError):compare(var(1),var(2),dimension=0)
        with self.assertRaises(ValueError):adjacent_certificate((1,1),0)
        with self.assertRaises(ValueError):cut_certificate(3,3)
        with self.assertRaises(ValueError):imm_words(7)
        with self.assertRaises(ValueError):finite_shift_action(var(1),{():F(1)},-1)

if __name__=='__main__':unittest.main()
