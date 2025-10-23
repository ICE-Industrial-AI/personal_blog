---
title: "A Renaissance Drama: The Race to Solve the Cubic"
description: "We present a clear and self-contained derivation of the solution formulas for cubic equations in two stages: first the depressed cubic x^3+px+q=0, and then the general monic cubic x^3+ax^2+bx+c=0 via a Tschirnhaus shift. Along the way we justify each algebraic step, derive the resolvent quadratic, discuss how to choose cube-root branches consistently, and analyze the structure of real and complex roots via the discriminant. Brief historical notes situate the method in the work of Cardano, with connections to del Ferro and Tartaglia."
pubDate: "Oct 23 2025"
heroImage: "/personal_blog/aikn.webp"
badge: "Latest"
---


# A Renaissance Drama: The Race to Solve the Cubic
*Author: Christoph Würsch, Institute for Computational Engineering, ICE, OST*


## Cardano's Method for Cubic Equations

Girolamo Cardano (1501–1576) published the general solution of the cubic in *Ars Magna* (1545). The depressed-cubic case was known earlier to Scipione del Ferro (c. 1465–1526) and was independently rediscovered by Niccolò Tartaglia (1500–1557). Cardano, after learning of del Ferro's prior discovery, included the method in his treatise and credited del Ferro. The modern streamlined derivation below follows the spirit of Cardano's presentation.

The story of the cubic formula isn't one of quiet contemplation in a university library; it's a tale of secrets, ambition, betrayal, and a dramatic intellectual duel fought in the public squares of 16th-century Italy.

<div align="center">
    <img src="/personal_blog/Girolamo-Cardano.jpeg" width="100%">
    <figcaption>Girolamo Cardano (1501-1576)</figcaption>
</div>

Our story begins in Bologna with **Scipione del Ferro**, a mathematics professor with a secret. In an era where academic reputations were made and broken in public challenges, a unique discovery was a weapon to be guarded. Del Ferro had found the holy grail of his time: a method for solving the "depressed" cubic equation, those of the form $x^3 + mx = n$. For years, he kept this powerful knowledge to himself, a secret key that could unlock problems no one else could solve. Only on his deathbed, around 1526, did he pass it on, whispering the formula to his student, the mediocre but ambitious **Antonio Fior**. Fior now possessed a treasure he hadn't earned, a single, powerful trick he believed would make him famous.

### The Challenger from the Gutter

Meanwhile, in Venice, a brilliant, self-taught mathematician named **Niccolò Fontana** was making a name for himself. Scarred on the face by a French soldier's sword as a child, the wound left him with a permanent stammer, earning him the nickname **"Tartaglia"** (the Stammerer), which he defiantly adopted. Tartaglia was a mathematical prodigy who rose from poverty to become one of Italy's most respected problem-solvers.

<div align="center">
    <img src="/personal_blog/Niccolo_Tartaglia.jpg" width="50%">
    <figcaption>Nicolò Tartaglia (1500-1557)</figcaption>
</div>

In 1535, news of Tartaglia's prowess reached Fior, who, seeing a chance for glory, made a fatal mistake: he challenged the Stammerer to a public mathematics contest. The rules were simple: each man would set 30 problems for the other. The winner would take home the prize money and, more importantly, the glory. Fior, confident in his secret weapon, crafted all his problems based on the $x^3 + mx = n$ form, certain that Tartaglia would be unable to solve them.

Tartaglia, receiving the list of problems, realized they were all of the same, unsolved type. Panic set in. With the clock ticking, he threw himself at the problem with ferocious intensity. In a stunning display of genius, on the night of February 13, 1535, just days before the deadline, Tartaglia had a breakthrough. He rediscovered the solution for himself! He swiftly solved all 30 of Fior's problems in a matter of hours. When the day of reckoning came, Tartaglia was triumphant. Fior, on the other hand, couldn't solve a single one of the varied problems Tartaglia had set for him. His one-trick pony had failed, and he faded into historical obscurity.

### The Oath and the Betrayal

Tartaglia's stunning victory made him a celebrity. News of his secret method reached Milan and the ears of the brilliant, eccentric, and insatiably curious polymath **Gerolamo Cardano**. A renowned physician, astrologer, and compulsive gambler, Cardano was also a mathematician of the first rank and was writing a comprehensive book on algebra, *Ars Magna* ("The Great Art"). The cubic solution was the one missing piece.

Cardano begged Tartaglia to share the secret. Tartaglia, wary and protective of his hard-won knowledge, repeatedly refused. But Cardano was persistent. He flattered Tartaglia, invited him to Milan, and promised him patronage and introductions. Finally, he cornered Tartaglia and swore a sacred oath:

> *"I swear to you, by God's holy Gospels, and as a true man of honour, not only never to publish your discoveries, if you teach them to me, but I promise you, and I pledge my faith as a true Christian, to note them down in code, so that after my death no one will be able to understand them."*

Moved by the solemnity of the oath, Tartaglia relented. He revealed his solution, possibly in the form of a cryptic poem, to keep it from being easily understood.

### The Great Art and the Aftermath

Cardano was no Fior. He not only understood the method but, with the help of his brilliant young student **Lodovico Ferrari**, mastered and expanded it. Together, they figured out how to solve *all* cubic equations, including those with an $x^2$ term. During this work, Cardano stumbled upon a startling implication: the formula sometimes required taking the square root of a negative number to arrive at a real answer. These were the first whispers of **complex numbers**.

Years passed. Cardano, honoring his oath, kept the secret. But then, on a trip to Bologna, he and Ferrari made a shocking discovery while examining del Ferro's old notebooks. There, plain as day, was the solution to the depressed cubic. Del Ferro had discovered it *before* Tartaglia.

Cardano saw his chance. His oath, he reasoned, bound him not to publish *Tartaglia's* work. But this was del Ferro's work, discovered earlier! Feeling absolved, in 1545, he published *Ars Magna*. In it, he presented the complete solution to the cubic and quartic (solved by Ferrari). He gave full credit to both del Ferro for the first discovery and Tartaglia for his independent rediscovery.

But for Tartaglia, it was the ultimate betrayal. His name was in the book, but the glory belonged to Cardano. Enraged, he launched a vicious public feud, accusing Cardano of theft and oath-breaking. Cardano himself refused to engage, letting his fiercely loyal student, Ferrari, fight in his place. Another public contest was arranged, this time between Tartaglia and Ferrari. The younger, more agile mind of Ferrari won the day. Tartaglia, defeated and humiliated, lost his prestige and died a few years later, impoverished and bitter.

Cardano's name is the one forever attached to the method, a testament to a dramatic and bitter chapter in the history of science, where genius, secrecy, and ambition collided to give the world one of its great mathematical treasures.

> We present a clear and self-contained derivation of the solution formulas for cubic equations in two stages: first the depressed cubic $x^3+px+q=0$, and then the general monic cubic $x^3+ax^2+bx+c=0$ via a Tschirnhaus shift. Along the way we justify each algebraic step, derive the resolvent quadratic, discuss how to choose cube-root branches consistently, and analyze the structure of real and complex roots via the discriminant. Brief historical notes situate the method in the work of Cardano, with connections to del Ferro and Tartaglia.

## The Depressed Cubic $x^3+px+q=0$

First, let's tackle a simplified version of the cubic equation called the **depressed cubic**. This is a cubic equation that is missing the $x^2$ term. It has the general form:

$$x^3 + p\,x + q \;=\; 0$$

Here, $p$ and $q$ are known numbers, which can be real or complex. The entire strategy hinges on a brilliantly clever substitution.

### Step 1: The substitution $x=u+v$

The core idea is to replace the single unknown variable $x$ with the sum of two new unknown variables, $u$ and $v$. So, we set $x = u+v$. This might seem strange—why make the problem more complicated by introducing *two* unknowns instead of one? The reason is that this gives us extra flexibility. We can later impose a convenient condition on $u$ and $v$ that will make the equation much simpler.

Let's substitute $x=u+v$ into our depressed cubic equation:

$$(u+v)^3 + p(u+v) + q = 0$$

Expanding the $(u+v)^3$ term gives us:

$$(u^3+3u^2v + 3uv^2+v^3) + p(u+v) + q = 0$$

Now, let's group the terms a bit differently. We can factor $3uv$ out of the middle two terms of the expansion:

$$u^3+v^3 + 3uv(u+v) + p(u+v) + q = 0$$

This is where the magic happens. We have two terms that contain $(u+v)$. Let's factor that out:

$$u^3+v^3 + (3uv + p)(u+v) + q = 0$$

Now we use the extra flexibility we gained from introducing two variables. We can **impose a condition** on $u$ and $v$ that simplifies this equation. Look at the term $(3uv + p)$. If we could make that whole term equal to zero, the equation would become much simpler! So, let's do exactly that. We will require that $u$ and $v$ satisfy the following constraint:

$$3uv + p = 0 \quad\Longleftrightarrow\quad uv = -\frac{p}{3}$$

By enforcing this condition, the middle part of our cubic equation simply vanishes! The equation now simplifies dramatically to:

$$u^3 + v^3 + q = 0 \quad\Longleftrightarrow\quad u^3+v^3 = -q$$

So, by substituting $x=u+v$ and imposing the condition $uv = -p/3$, we've transformed our original problem into finding two numbers, $u$ and $v$, that satisfy two new equations:
1.  $u^3+v^3 = -q$
2.  $uv = -p/3$

### Step 2: The resolvent quadratic for $u^3$ and $v^3$

Let's look at the two conditions we have. If we cube the second condition, we get $(uv)^3 = (-p/3)^3$, which is $u^3v^3 = -p^3/27$.

Now, let's make another small substitution to make things even clearer. Let $s = u^3$ and $t = v^3$. Our two conditions become:

1.  **Sum:** $s + t = -q$
2.  **Product:** $st = -p^3/27$

This is a classic problem! If you know the sum and product of two numbers ($s$ and $t$), you can find them by forming a quadratic equation. A quadratic equation with roots $s$ and $t$ can be written as $(z-s)(z-t) = 0$. Expanding this gives $z^2 - (s+t)z + st = 0$.

We already know $(s+t)$ and $st$, so we can plug them right in:

$$z^2 - (-q)\,z + \left(-\frac{p^3}{27}\right) \;=\; 0$$

This simplifies to what is known as the **resolvent quadratic equation**:

$$z^2 + q\,z - \frac{p^3}{27} \;=\; 0$$

We can solve this for $z$ using the quadratic formula, $z = \frac{-B \pm \sqrt{B^2-4AC}}{2A}$. In our case, $A=1$, $B=q$, and $C=-p^3/27$. The two solutions for $z$ will be our values for $s$ and $t$.

$$s, t = \frac{-q \pm \sqrt{q^2 - 4(1)(-p^3/27)}}{2} = -\frac{q}{2} \pm \sqrt{\frac{q^2}{4} + \frac{p^3}{27}}$$

So, the two solutions are:
$$s = -\frac{q}{2} + \sqrt{\left(\frac{q}{2}\right)^2 + \left(\frac{p}{3}\right)^3}, \qquad t = -\frac{q}{2} - \sqrt{\left(\frac{q}{2}\right)^2 + \left(\frac{p}{3}\right)^3}$$

Let's define the expression inside the square root as $D$, which is closely related to the discriminant:
$$D \;:=\; \left(\frac{q}{2}\right)^2 + \left(\frac{p}{3}\right)^3$$
With this definition, our solutions for $s$ and $t$ become much cleaner: $s = -q/2 + \sqrt{D}$ and $t = -q/2 - \sqrt{D}$.

### Step 3: Taking matched cube roots and solving for $x$

We've found $s$ and $t$. Remember that we defined $s=u^3$ and $t=v^3$. To find $u$ and $v$, we just need to take the cube roots:

$$u = \sqrt[3]{s} = \sqrt[3]{-\frac{q}{2}+\sqrt{D}}$$
$$v = \sqrt[3]{t} = \sqrt[3]{-\frac{q}{2}-\sqrt{D}}$$

Since our original substitution was $x=u+v$, the solution for $x$ appears to be:

$$x \;=\; \sqrt[3]{-\frac{q}{2}+\sqrt{D}} \;+\; \sqrt[3]{-\frac{q}{2}-\sqrt{D}}$$

This is Cardano's famous formula for the depressed cubic! However, there's a very important subtlety here. Any non-zero complex number has **three** distinct cube roots. So when we write $\sqrt[3]{s}$, there are three possible values for $u$. Similarly, there are three possible values for $v$. This gives $3 \times 3 = 9$ possible combinations for the pair $(u, v)$. Which one is correct?

This is where our constraint from Step 1, $uv = -p/3$, becomes critical again. We must choose the cube root for $u$ and the cube root for $v$ such that their product is exactly $-p/3$. This requirement "matches" the roots and reduces the nine possible pairs to just three valid ones, which correspond to the three roots of the cubic equation.

**Branch matching with cubic roots.**
To understand this matching, we use the complex roots of unity. The three cube roots of 1 are $1$, $\omega = e^{2\pi i/3} = -\frac{1}{2}+i\frac{\sqrt{3}}{2}$, and $\omega^2 = e^{4\pi i/3} = -\frac{1}{2}-i\frac{\sqrt{3}}{2}$. Notice that $1+\omega+\omega^2=0$.

If $u_0$ is one specific cube root of $s$, then the other two are $\omega u_0$ and $\omega^2 u_0$. Similarly, if $v_0$ is one cube root of $t$, the others are $\omega v_0$ and $\omega^2 v_0$.

Let's say we pick a pair $(u_0, v_0)$ such that their product $u_0 v_0 = -p/3$. What about the other combinations?
* $(\omega u_0) \times (\omega v_0) = \omega^2 (u_0 v_0)$ --- This product is wrong.
* $(\omega u_0) \times (\omega^2 v_0) = \omega^3 (u_0 v_0) = 1 \times (u_0 v_0)$ --- This product is right!
* $(\omega^2 u_0) \times (\omega v_0) = \omega^3 (u_0 v_0) = 1 \times (u_0 v_0)$ --- This is also right.

To keep the product $uv$ constant, if we multiply $u_0$ by a root of unity (say, $\omega^k$), we must multiply $v_0$ by its inverse ($\omega^{-k}$). This gives us the three valid pairs that satisfy the constraint:
$$(u_k,v_k) := \big(\omega^k u_0,\ \omega^{-k} v_0\big), \qquad k=0,1,2$$
The three sums $x_k=u_k+v_k$ give the three roots of the cubic equation.

### Step 4: The discriminant and the nature of roots

The value $D = (q/2)^2 + (p/3)^3$ tells us about the nature of the roots. It's directly related to the polynomial **discriminant**, $\Delta$, by the formula:

$$\Delta \;=\; -4p^3 - 27q^2 \;=\; -108\,D$$

The sign of $\Delta$ (or equivalently, the sign of $D$) separates the solutions into three cases for equations with real coefficients $p$ and $q$:

* **$D > 0$ ($\Delta < 0$)**: In this case, $\sqrt{D}$ is a real number. The two cube roots in the formula are of real numbers. This gives **one real root** and **two complex conjugate roots**. Cardano's formula works perfectly here and directly gives you the real root.

* **$D = 0$ ($\Delta = 0$)**: Here, $\sqrt{D}=0$, so $u=v=\sqrt[3]{-q/2}$. The equation has **multiple real roots**. Specifically, if $p$ and $q$ are not both zero, there is a double root and a simple root. If $p=q=0$, there is a triple root at $x=0$.

* **$D < 0$ ($\Delta > 0$)**: This is the most interesting and historically challenging case, known as the ***casus irreducibilis*** (the irreducible case). Here, $\sqrt{D}$ is an imaginary number. This means that to find the roots, you must take the cube root of complex numbers. The paradox is that in this case, the equation has **three distinct real roots**. The formula forces you to go through the complex numbers to find real solutions, which baffled mathematicians for centuries.

### Step 5: Trigonometric (all-real) representation when $D<0$

For the *casus irreducibilis*, where we have three real roots but the formula involves complex numbers, there is a more practical way to find the roots using trigonometry. This method avoids complex radicals. If $p<0$ and $D<0$, we can make the substitution:

$$x \;=\; 2\sqrt{-\frac{p}{3}}\;\cos\theta$$

Plugging this into the cubic $x^3+px+q=0$ and using the trigonometric identity $\cos(3\theta)=4\cos^3\theta-3\cos\theta$ leads to:

$$2\left(-\frac{p}{3}\right)^{3/2} \cos(3\theta) + q = 0$$

We can then solve for $\cos(3\theta)$:

$$\cos(3\theta) \;=\; -\frac{q}{2}\,\sqrt{-\frac{27}{p^3}}$$

The value on the right-hand side will be between -1 and 1 precisely when $D \le 0$. We can find the angle $3\theta$ by taking the arccosine. Since cosine is a periodic function, there are multiple angles that work, which is how we find all three roots. The three solutions for $x$ are given by:

$$
x_k \;=\; 2\sqrt{-\frac{p}{3}}\;
\cos\!\left(\frac{1}{3}\arccos\!\Bigl( -\frac{q}{2}\sqrt{-\frac{27}{p^3}}\Bigr) + \frac{2\pi k}{3}\right),
\qquad k=0,1,2
$$

This formula gives all three real roots without any intermediate complex numbers.

**Special cases.**
If $p=0$, then $x^3+q=0 \Rightarrow x=-\sqrt[3]{q}$.
If $q=0$, then $x(x^2+p)=0 \Rightarrow x=0$ or $x=\pm\sqrt{-p}$.




## The General Cubic $x^3+ax^2+bx+c=0$

Now, what if our cubic equation has an $x^2$ term?
$$x^3 + a\,x^2 + b\,x + c \;=\; 0$$
The strategy is to transform this general cubic into a depressed cubic, which we already know how to solve!

### Step 1: Depress the cubic (kill the $x^2$-term)

We can eliminate the $x^2$ term by shifting the function horizontally. This is achieved with a substitution known as a **Tschirnhaus transformation**. We set:

$$x = y - \frac{a}{3}$$

Why this specific value? When you substitute this into the general cubic and expand the terms, the terms involving $y^2$ will perfectly cancel out. Let's see it:
$(y-a/3)^3 + a(y-a/3)^2 + ...$
The first term gives $-3y^2(a/3) = -ay^2$.
The second term gives $+ay^2$.
These two terms cancel, eliminating the squared term!

Substituting $x = y-a/3$ fully and collecting terms gives us a new depressed cubic in the variable $y$:

$$y^3 + p\,y + q \;=\; 0$$

where the new coefficients $p$ and $q$ are defined in terms of the old ones ($a,b,c$):

$$p = b - \frac{a^2}{3}, \quad q = \frac{2a^3}{27} - \frac{ab}{3} + c$$

### Step 2: Apply Cardano to the depressed form

We now have a depressed cubic $y^3 + py + q = 0$. We can solve this for $y$ using the exact method we developed in the first section. We calculate our value $D$ using the new $p$ and $q$:

$$D = \left(\frac{q}{2}\right)^2 + \left(\frac{p}{3}\right)^3$$
Let $s = -q/2 + \sqrt{D}$ and $t = -q/2 - \sqrt{D}$. Then, by taking properly matched cube roots $u=\sqrt[3]{s}$ and $v=\sqrt[3]{t}$ such that $uv=-p/3$, we find the solutions for $y$:

$$y = u + v = \sqrt[3]{-\frac{q}{2}+\sqrt{D}} + \sqrt[3]{-\frac{q}{2}-\sqrt{D}}$$

Finally, we must remember that we solved for $y$, not our original variable $x$. To get the solution for $x$, we just undo the substitution from Step 1: $x = y - a/3$. This gives the complete formula for the general cubic equation:

$$x \;=\; -\frac{a}{3} \;+\; \sqrt[3]{-\frac{q}{2}+\sqrt{D}} \;+\; \sqrt[3]{-\frac{q}{2}-\sqrt{D}}$$
where $p$ and $q$ are calculated from $a,b,c$ as shown in Step 1.

### Step 3: All three roots via cubic roots of unity

Just as before, we can find all three roots. Let $u_0$ and $v_0$ be principal cube roots of $s$ and $t$ that satisfy $u_0v_0=-p/3$. The three solutions for $y$ are:

$$y_k = \omega^{\,k}u_0 + \omega^{-k} v_0,\qquad k=0,1,2$$
The three solutions for $x$ are then found by undoing the shift for each $y_k$:

$$x_k \;=\; y_k - \frac{a}{3},\qquad k=0,1,2$$



## Discriminant and Complex-Root Analysis for the General Cubic

The horizontal shift $x = y - a/3$ does not change the nature of the roots (e.g., how many are real vs. complex). Therefore, the discriminant of the depressed cubic $y^3+py+q=0$ tells us everything we need to know about the roots of the original cubic $x^3+ax^2+bx+c=0$.

We use the same discriminant, calculated using the $p$ and $q$ from the depressed form:
$$\Delta \;=\; -4p^3 - 27q^2 \;=\; -108\,D$$
* **$\Delta > 0$ ($D < 0$)**: three distinct real roots.
* **$\Delta = 0$ ($D = 0$)**: a multiple root (either a triple root, or one simple + one double root). All roots are real.
* **$\Delta < 0$ ($D > 0$)**: one real root and two nonreal complex-conjugate roots.



## Summary

Cardano's method is a beautiful and powerful algorithm for solving any cubic equation. The overall strategy is:

1.  **Depress the Cubic:** If the equation has an $x^2$ term, use the substitution $x = y - a/3$ to eliminate it. This reduces the problem to solving a simpler "depressed" cubic.
2.  **The $u+v$ Substitution:** For the depressed cubic $y^3+py+q=0$, substitute $y = u+v$. This allows you to impose the condition $uv = -p/3$, which simplifies the equation immensely.
3.  **Solve the Resolvent Quadratic:** The problem is now reduced to finding $u^3$ and $v^3$. Their sum is $-q$ and their product is $-p^3/27$. These two values are the roots of a simple quadratic equation.
4.  **Take Matched Cube Roots:** Find $u$ and $v$ by taking the cube roots. Crucially, you must choose the roots that satisfy the condition $uv = -p/3$. Using the complex roots of unity allows you to find all three valid pairs, giving the three solutions for $y$.
5.  **Undo the Shift:** Once you have the solutions for $y$, use the relation $x = y - a/3$ to find the solutions to your original equation.

The discriminant $\Delta=-4p^3-27q^2$ reveals the nature of the roots (real, complex, or multiple) without having to solve the entire equation.

**Further reading:** G. Cardano, *Ars Magna*, 1545 (English trans. T. Richard Witmer, 1968); R. Bombelli, *L'Algebra*, 1572, for early systematic use of complex numbers in the cubic.