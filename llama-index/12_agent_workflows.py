"""
What is a workflow?

A workflow is an event-driven, step-based way to control the execution flow of an application.

Your application is divided into sections called Steps which are triggered by Events, and themselves 
emit Events which trigger further steps. By combining steps and events, you can create arbitrarily 
complex flows that encapsulate logic and make your application more maintainable and easier to understand. 
A step can be anything from a single line of code to a complex agent. They can have arbitrary inputs 
and outputs, which are passed around by Events.

Why workflows?

As generative AI applications become more complex, it becomes harder to manage the flow of data and 
control the execution of the application. Workflows provide a way to manage this complexity by breaking 
the application into smaller, more manageable pieces.

Other frameworks and LlamaIndex itself have attempted to solve this problem previously with directed 
acyclic graphs (DAGs) but these have a number of limitations that workflows do not:

- Logic like loops and branches needed to be encoded into the edges of graphs, which made them hard to read and understand.
- Passing data between nodes in a DAG created complexity around optional and default values and which parameters should be passed.
- DAGs did not feel natural to developers trying to developing complex, looping, branching AI applications.

The event-based pattern and vanilla python approach of Workflows resolves these problems.

Note:
The Workflows library can be installed standalone, via `pip install llama-index-workflows`. However,
`llama-index-core` comes with an installation of Workflows included.

In order to maintain the `llama_index` API stable and avoid breaking changes, when installing 
`llama-index-core` or the `llama-index` umbrella package, Workflows can be accessed with the 
import path `llama_index.core.workflow`.
"""
