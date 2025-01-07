1, Why did you implement rootBA on Basalt rather than DSO, another famous work from TUM? Since rootBA is designed for solving large-scale BA, and DSO maintains a much larger BA.

The technique we presented in the paper is suitable for different approaches. We just found it to be particularly advantageous for a setup like in Basalt. One can also try it in cases like DSO, but there may be some challenges, as you have found out. We didn't try it for this project.

2, Actually I implemented rootBA on a direct visual odometry, BUT it ended up with about 7% more CPU occupation, worse accuracy and robustness. HOWEVER basalt shows remarkable performance no matter on runtime or accuracy. Is it the answer of question 1?

Hard to say without further information. The DSO code is quite complex and not particularly well-documented. AFAIK it employs a lot of computational tricks to make things fast. It might be hard to have all of that in mind when making changes.
One thing to keep in mind is that in DSO every observation has an 8-dimensional residual, while in Basalt it's 2-dimensional. This means the Jacobians have 8 rows per observation (expect for the camera that hosts the point, which doesn't have a residual), while in Basalt it's only 2 rows per observation. Larger Jacobians also mean that the QR factorization is more expensive, so the runtime tradeoff might be different.
However, I would expect superior robustness / numerical stability also for DSO if you use nullspace marginalization compared to the Schur complement, in particular in single precision.

3, Does Basalt's accuracy benefit from stereographic projection rather than rootBA?

I'm not sure what you mean. The use of stereographic projection in the landmark parameterization is independent from the ideas of square root bundle adjustment / square root marginalization.

4, "...to keep the number of optimiziation variables small, we parametrize the bearing vector in 3D space using a minimal representation, which is two-dimensional", says the paper of Basalt.  But actually the optimiziation dimension is still three. Could you please make a further explanation?

This just refers to 2-dimensions for the landmark direction. Alternative non-minimal parameterizations of the direction could be a bearing vector or a rotation matrix. Together with inverse depth it's still 3 dimensional, that is correct.
