{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<div class=\"alert alert-info\">\n",
    "    <h1 align=\"center\">Normal Equation</h1>\n",
    "    <h3 align=\"center\"> Machine Learning (Fall 2018)</h3>\n",
    "    <h5 align=\"center\">Seyed Naser Razavi [ML2018](http://www.snrazavi.ir/ml-2018/)</h5>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Introduction"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src='imgs/normal_equation_detail.png' width='75%'>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Goal.** Given $X$ and $y$, our goal is to find $\\theta$ such that: "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "$$X \\theta = y$$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "1. To solve the above equation for $\\theta$, first we multiply both sides by $X^T$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "$$X^T X\\theta = X^T y$$ "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "2. Now, we can compute the **inverse** matrix of $X^T X$ and multiply it on both sides of the above equation."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "$$(X^T X)^{-1} (X^T X) \\theta = (X^T X)^{-1} X^T y$$ "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Hence, we obtatin"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "$$\\theta = (X^T X)^{-1} X^T y = X^{+} y$$ "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<div class=\"alert alert-info\">\n",
    "    <strong>Pseudo-inverse:</strong> In the above equation, $X^+$ is called pseudo-inverse.\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Implentation in Python"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from numpy.linalg import pinv  # pseudo-inverse\n",
    "\n",
    "np.set_printoptions(precision=2, sign=' ', suppress=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Inputs (design matrix)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "X = np.array([[1, 2104, 5, 1, 45],\n",
    "              [1, 1416, 3, 2, 40],\n",
    "              [1, 1534, 3, 2, 30],\n",
    "              [1,  852, 2, 1, 36]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[   1 2104    5    1   45]\n",
      " [   1 1416    3    2   40]\n",
      " [   1 1534    3    2   30]\n",
      " [   1  852    2    1   36]]\n"
     ]
    }
   ],
   "source": [
    "print(X)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Outputs (targets)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "y = np.array([[460.],\n",
    "              [232.],\n",
    "              [315.],\n",
    "              [178.]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 460.]\n",
      " [ 232.]\n",
      " [ 315.]\n",
      " [ 178.]]\n"
     ]
    }
   ],
   "source": [
    "print(y)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Solving Normal Equation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "$$\\theta = (X^T X)^{-1} X^T y = X^{+} y$$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# ??np.linalg.pinv  # getting help"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "theta = pinv(X) @ y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[188.4 ],\n",
       "       [  0.39],\n",
       "       [-56.14],\n",
       "       [-92.97],\n",
       "       [ -3.74]])"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "theta"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 460.]\n",
      " [ 232.]\n",
      " [ 315.]\n",
      " [ 178.]]\n"
     ]
    }
   ],
   "source": [
    "print(X @ theta)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 460.]\n",
      " [ 232.]\n",
      " [ 315.]\n",
      " [ 178.]]\n"
     ]
    }
   ],
   "source": [
    "print(y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[False]\n",
      " [False]\n",
      " [False]\n",
      " [False]]\n"
     ]
    }
   ],
   "source": [
    "print(X @ theta == y)      # Wrong way to campare to float matrices"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.allclose(X @ theta, y)  # correct way for comparing numpy matrices"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Better option:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The matrix $X^T X$ may not be invertible."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "theta = pinv(X.T @ X) @ X.T @ y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[188.4 ],\n",
       "       [  0.39],\n",
       "       [-56.14],\n",
       "       [-92.97],\n",
       "       [ -3.74]])"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "theta"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.allclose(X @ theta, y)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Inverse and Pseudo-Inverse"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<div class=\"alert alert-info\">\n",
    "    <strong>Inverse: ($A^{-1}$)</strong>\n",
    "    $$A A^{-1} = A^{-1} A = I$$\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<div class=\"alert alert-info\">\n",
    "    <strong>Pseudo-inverse: ($A^{+}$)</strong>\n",
    "    $$A^{+} = (A^T A)^{-1} A^T$$\n",
    "</div>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from numpy.linalg import inv, pinv"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "A = np.random.randn(3, 3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0.38, -1.31,  0.86],\n",
       "       [-1.1 , -0.88, -0.7 ],\n",
       "       [-0.5 ,  0.11, -0.67]])"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "A"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 2.39, -2.8 ,  5.95],\n",
       "       [-1.38,  0.62, -2.41],\n",
       "       [-2.02,  2.19, -6.33]])"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "inv(A)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 2.39, -2.8 ,  5.95],\n",
       "       [-1.38,  0.62, -2.41],\n",
       "       [-2.02,  2.19, -6.33]])"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pinv(A)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Check for equality"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.allclose(inv(A), pinv(A))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<div class=\"alert alert-success\">\n",
    "    <strong>Question:</strong> So, what is the difference between `inv` and `pinv`\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<div class=\"alert alert-info\">\n",
    "    <strong>Theorem:</strong> If matrix $A$ is **non-singular**, then $A^{-1} = A^{+}$.\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Proof.**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "$$A^{+} = (A^T A)^{-1} A^T = (A^{-1} (A^T)^{-1}) A^T = A^{-1} ((A^T)^{-1} A^T) = A^{-1} I = A^{-1}$$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "But, what if the matrix is **singular**?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "A = np.array([[1, 2, 3], \n",
    "              [4, 5, 6], \n",
    "              [2, 4, 6]])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now, if we try to compute `inv(A)`, we will get a `LinAlgError`! (try it yourself)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src='imgs/linAlgError-SingularMatrix.png' width='76%'>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[-0.19,  0.44, -0.38],\n",
       "       [-0.02,  0.11, -0.04],\n",
       "       [ 0.14, -0.22,  0.29]])"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pinv(A)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "heading_collapsed": true
   },
   "source": [
    "## Another Example"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true,
    "hidden": true
   },
   "outputs": [],
   "source": [
    "# inputs (or design matrix)\n",
    "X = np.array([[1, 2104, 5, 1, 45],\n",
    "              [1, 1416, 3, 2, 40],\n",
    "              [1, 1534, 3, 2, 30],\n",
    "              [1,  852, 2, 1, 36], \n",
    "              [1, 3000, 4, 1, 38]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true,
    "hidden": true
   },
   "outputs": [],
   "source": [
    "# desired outputs\n",
    "y = np.array([[460.],\n",
    "              [232.],\n",
    "              [315.],\n",
    "              [178.], \n",
    "              [540.]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true,
    "hidden": true
   },
   "outputs": [],
   "source": [
    "theta = pinv(X.T @ X) @ X.T @ y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true,
    "hidden": true
   },
   "outputs": [],
   "source": [
    "theta"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true,
    "hidden": true
   },
   "outputs": [],
   "source": [
    "X @ theta"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true,
    "hidden": true
   },
   "outputs": [],
   "source": [
    "y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true,
    "hidden": true
   },
   "outputs": [],
   "source": [
    "np.allclose(X @ theta, y)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Psuedo-inverse properties (optional)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<div class=\"alert alert-info\">\n",
    "    <strong>Property 1:</strong>\n",
    "    $$(A^+)^+ = A$$\n",
    "</div>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.allclose(pinv(pinv(X)), X)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<div class=\"alert alert-info\">\n",
    "    <strong>Property 2:</strong>\n",
    "    $$A A^+ A = A$$\n",
    "</div>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.allclose(X @ pinv(X) @ X, X)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<div class=\"alert alert-info\">\n",
    "    <strong>Property 3:</strong>\n",
    "    $$A^+ A A^+ = A^+$$\n",
    "</div>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.allclose(pinv(X) @ X @ pinv(X), pinv(X))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
