# TinyGrad: An Autograd Engine
TinyGrad is an autograd engine built entirely using python3.14. The actual engine is inside of [value.py](value.py), with a Multi-Layer-Perceptron to train networks in [nn.py](nn.py).

## Qualities and Drawbacks
TinyGrad is truthful in what it is: a tiny autograd engine based on a very simple backpropagation algorithm using chain-rule.

It is minimal, which means it can be a great tool for teaching and learning neural networks.

However, coming in a compact form-factor has its obvious drawbacks: The training is done entirely on the Python runtime, therefore it is incredibly impractical for moderate-to-serious deep-learning. There is also no support for tensors, as the engine is built only to take scalar quantities as input.

That does not limit its correctess, though. Albeit very slow, it still leverages similar techniques used in industry-based machine/deep learning.

## Credits
While the engine is built solely by me, it takes big inspiration from AI researcher Andrej Karpathy and his 'micrograd' project.

Thank you for using TinyGrad.