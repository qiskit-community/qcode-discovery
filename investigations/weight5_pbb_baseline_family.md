# Fixed weight-five PBB baseline: proved statements and a counterexample

This note records what can currently be proved about the fixed PBB family

\[
 (A,B,C,D)=(1+x+x^2,\;1+y,\;x,\;1+y)
\]

over

\[
 R=\mathbb F_2[x,y]/(x^\ell-1,y^m-1).
\]

It is an investigation note, not manuscript text.  No general distance lower
bound is claimed here.

## Result

Assume \(\ell\geq 3\), \(3\mid\ell\), and \(m\geq2\).  The last hypothesis
keeps \(1\) and \(y\) distinct, so this is genuinely the stated weight-five
family.  Then

\[
 k=2\gcd(m,2)
 =\begin{cases}2,&m\text{ odd},\\4,&m\text{ even},\end{cases}
\]

and there are explicit nontrivial logical operators of weights \(2\ell/3\)
and \(m\).  Consequently,

\[
 \boxed{d\leq\min\!\left(\frac{2\ell}{3},m\right).}
\]

In particular, the formula \(d=2\ell/3\) is **not true under only the
hypothesis \(3\mid\ell\)**.  For example, \((\ell,m)=(6,2)\) has a
weight-two logical although \(2\ell/3=4\).  Exhausting all 72 weight-one
Paulis independently gives the exact counterexample `[[24,4,2]]`.

The catalogue instances all satisfy \(m\geq2\ell/3\), so this obstruction is
invisible in the sampled sequence.  In that restricted regime the construction
below proves \(d\leq2\ell/3\), but a matching general lower bound remains open.

## Stabilizer convention and commutativity

Write \(N=\ell m\), so the code has \(n=2N\) qubits, and let
\(f^*=f(x^{-1},y^{-1})\).  The two translated stabilizer spaces can be written

\[
 \begin{aligned}
 T(u)&=(uA,uB\mid uC,uD),\\
 Z(v)&=(0,0\mid vB^*,vA^*),\qquad u,v\in R.
 \end{aligned}
\]

The bottom rows commute with one another.  Top--bottom commutation is the
usual abelian BB cancellation.  For the top rows, the relevant group-ring
element is

\[
 AC^*+BD^*
 =(1+x+x^2)x^{-1}+(1+y)(1+y^{-1})
 =x^{-1}+1+x+y+y^{-1},
\]

which is fixed by \(*\).  Thus the displayed rows define a commuting
stabilizer code for every \(\ell\geq3\) and \(m\geq2\).

## Dimension proof

Let

\[
 R_x=\mathbb F_2[x]/(x^\ell-1),\qquad
 R_y=\mathbb F_2[y]/(y^m-1),
\]

so \(R\cong R_x\otimes R_y\).

For multiplication by \(f\) on \(\mathbb F_2[z]/(g)\), the kernel has
dimension \(\deg\gcd(f,g)\).  Since \(3\mid\ell\),

\[
 \dim\operatorname{Ann}_{R_x}(1+x+x^2)=2.
\]

The same is true for \(A^*=x^{-2}A\), since the monomial factor is a unit.
Also

\[
 \operatorname{Ann}_{R_y}(1+y)=\langle h\rangle,qquad
 h=1+y+\cdots+y^{m-1},
\]

and this annihilator has dimension one.  Again \(B^*=y^{-1}B\) has the same
annihilator.

The map \(T\) is injective because its third component is \(ux\), and \(x\)
is a unit.  Hence \(\dim T(R)=N\).  The kernel of \(Z\) is

\[
 \operatorname{Ann}(A^*)\cap\operatorname{Ann}(B^*)
 =\operatorname{Ann}_{R_x}(A^*)\otimes\langle h\rangle,
\]

which has dimension two.  Therefore \(\dim Z(R)=N-2\).

It remains to compute \(T(R)\cap Z(R)\).  If \(T(u)=Z(v)\), its X part gives
\(uA=uB=0\), so

\[
 u=q(x)h(y),\qquad q\in\operatorname{Ann}_{R_x}(A).
\]

Because \(D=B\), the remaining equations reduce to

\[
 vA^*=0,\qquad vB^*=ux.
\]

The image of multiplication by \(1+y\) on \(R_y\) is exactly the
even-coefficient-parity subspace: it is contained there and both spaces have
dimension \(m-1\).  The coefficient parity of \(h\) is \(m\bmod2\).
Consequently, \(h\in\operatorname{im}(B^*)\) exactly when \(m\) is even.
Multiplication by \(x\) preserves the two-dimensional annihilator of \(A\),
so

\[
 \dim(T(R)\cap Z(R))=
 \begin{cases}0,&m\text{ odd},\\2,&m\text{ even}.
 \end{cases}
\]

Thus

\[
 \begin{aligned}
 \operatorname{rank}S
 &=N+(N-2)-\dim(T(R)\cap Z(R)),\\
 k&=2N-\operatorname{rank}S
 =\begin{cases}2,&m\text{ odd},\\4,&m\text{ even},\end{cases}
 \end{aligned}
\]

as claimed.  This proof does not assume square-free factorization, so it also
covers even \(\ell\) or \(m\), where the cyclic quotient can be non-semisimple.

## Horizontal logical of weight \(2\ell/3\)

Put \(t=\ell/3\) and define

\[
 Q=(1+x)\sum_{j=0}^{t-1}x^{3j}
  =\sum_{j=0}^{t-1}(x^{3j}+x^{3j+1}).
\]

All \(2t\) displayed monomials are distinct.  Moreover,

\[
 AQ=(1+x+x^2)(1+x)\sum_{j=0}^{t-1}x^{3j}
    =(1+x^3)\sum_{j=0}^{t-1}x^{3j}
    =1+x^\ell=0
\]

in \(R\).  Consider

\[
 L_x=(0,Q\mid0,Q).
\]

This is \(Y\) on the support of \(Q\) in the second qubit block, so its Pauli
weight is \(2t=2\ell/3\).  It commutes with the bottom stabilizers because
\(A^*Q=x^{-2}AQ=0\).  It commutes with the top stabilizers because its X and Z
contributions in the second block cancel when \(D=B\).

It is not a stabilizer.  If \(L_x=T(u)+Z(v)\), comparison of X components
would require \(uB=Q\).  Apply the well-defined homomorphism
\(R\to R_x\) given by \(y\mapsto1\).  Its left side is zero because
\(B(x,1)=0\), whereas its right side is the nonzero polynomial \(Q\), a
contradiction.

## Vertical logical of weight \(m\)

Using the same \(h=1+y+\cdots+y^{m-1}\), define

\[
 L_y=(0,0\mid0,h).
\]

It has Pauli weight \(m\).  It commutes with every bottom row because both are
pure Z.  Its only pairing with a top row is through the second X block, and
\(Bh=(1+y)h=1+y^m=0\), so it is in the normalizer.

It is also nontrivial.  If \(L_y=T(u)+Z(v)\), its zero X component would give
\(uA=uB=0\).  Since \(D=B\), this also gives \(uD=0\), and comparison of the
second Z component would force

\[
 vA^*=h.
\]

This is impossible.  Expand \(v=\sum_j v_j(x)y^j\).  Because \(A^*\) depends
only on \(x\), the displayed equality would require \(v_jA^*=1\) in \(R_x\)
for every \(j\).  But \(A^*\) is a zero divisor: it has a nonzero annihilator
when \(3\mid\ell\).  A zero divisor cannot have a multiplicative inverse.
Therefore \(L_y\) is a logical operator.

This second construction is the missing obstruction to the unqualified
distance formula.

## Checked cases

The independent checker in
[`scripts/verify_weight5_pbb_baseline_family.py`](../scripts/verify_weight5_pbb_baseline_family.py)
builds binary symplectic rows directly as Python integers.  It does not call
qLDPC or the repository PBB constructor.  It checks pairwise commutativity,
the ranks used above, normalizer membership, stabilizer nonmembership, and the
two logical weights.  Focused regression tests are in
[`tests/test_weight5_pbb_baseline_family.py`](../tests/test_weight5_pbb_baseline_family.py).

The nine exact catalogue instances and one unresolved instance are:

| \((\ell,m)\) | \((n,k)\) | recorded distance | \(2\ell/3\) | \(m\) |
|---:|---:|---:|---:|---:|
| \((6,6)\) | \((72,4)\) | \(4\) exact | 4 | 6 |
| \((6,9)\) | \((108,2)\) | \(4\) exact | 4 | 9 |
| \((6,10)\) | \((120,4)\) | \(4\) exact | 4 | 10 |
| \((9,8)\) | \((144,4)\) | \(6\) exact | 6 | 8 |
| \((9,9)\) | \((162,2)\) | \(6\) exact | 6 | 9 |
| \((12,9)\) | \((216,2)\) | \(8\) exact | 8 | 9 |
| \((12,12)\) | \((288,4)\) | \(8\) exact | 8 | 12 |
| \((15,12)\) | \((360,4)\) | \(10\) exact | 10 | 12 |
| \((15,14)\) | \((420,4)\) | \(10\) exact | 10 | 14 |
| \((18,12)\) | \((432,4)\) | \(5\leq d\leq12\) | 12 | 12 |

Every row lies in the regime \(m\geq2\ell/3\).  The following valid
weight-five instances lie outside it and directly contradict the unrestricted
formula:

| \((\ell,m)\) | \((n,k)\) | old prediction | proved vertical bound |
|---:|---:|---:|---:|
| \((6,2)\) | \((24,4)\) | \(d=4\) | \(d\leq2\) (in fact \(d=2\)) |
| \((6,3)\) | \((36,2)\) | \(d=4\) | \(d\leq3\) (checked \(d=3\)) |
| \((9,2)\) | \((36,4)\) | \(d=6\) | \(d\leq2\) |
| \((12,6)\) | \((144,4)\) | \(d=8\) | \(d\leq6\) |
| \((18,2)\) | \((72,4)\) | \(d=12\) | \(d\leq2\) (in fact \(d=2\)) |

The exact checks for \((6,2)\), \((6,3)\), and \((18,2)\) exhaust all
lower-weight Pauli supports and types.  The last example has \(n=72\), so the
counterexample is not merely below the manuscript's length range.  Exact small
cases are useful evidence for the corrected guess
\(d=\min(m,2\ell/3)\), but they are not a proof of that guess.

## Safe manuscript-level conclusion

The dimension part of the former empirical rule can be promoted to a theorem.
The distance part should instead be split into:

1. the proved upper bound \(d\leq\min(m,2\ell/3)\), with both explicit
   logicals shown;
2. the exact finite data in the table above; and
3. at most a restricted conjecture that \(d=2\ell/3\) when
   \(m\geq2\ell/3\), clearly labelled as unproved.

A stronger and cleaner conjecture suggested by the two constructions is
\(d=\min(m,2\ell/3)\), but it should not be stated as a result without a
general lower-bound argument.
