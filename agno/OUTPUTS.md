# Agno Example Outputs

Captured outputs from running all 22 examples against agno v3.0.9 with `gpt-4o-mini`.

> These outputs may vary between runs due to LLM non-determinism. The structure and tool invocations should remain consistent.

---

## 00_hello_world.py

```
╭──────────────────────────────────────────────────────────────────────────────╮
│ The phrase "hello, world" originates from a simple program written in the C  │
│ programming language, used as a beginner's tutorial to demonstrate basic     │
│ syntax and output.                                                           │
╰──────────────────────────────────────────────────────────────────────────────╯
```

---

## 01_agent_with_tools.py

```
╭────────────────────────────────────────╮
│ ### Current Weather in Lisbon          │
│ - **Condition:** Sunny                 │
│ - **Temperature:** 25°C                │
│ - **Wind:** Light breeze from the west │
│                                        │
│ ### Current Time in Lisbon             │
│ - **Time:** 14:30 (UTC+1)              │
╰────────────────────────────────────────╯
```

---

## 02_async_agent.py

```
=== Query 1 ===
╭──────────────────────────────────────────────────────────────────────────────╮
│ ### Latest Updates in Python (2023)                                          │
│                                                                              │
│ - **Python 3.13 Released**: The latest version introduces significant        │
│ **performance improvements** along with a variety of **new features**. This  │
│ release focuses on enhancing the overall efficiency and usability of the     │
│ Python programming language.                                                 │
│                                                                              │
│ For more detailed information on specific features or changes in Python      │
│ 3.13, feel free to ask!                                                      │
╰──────────────────────────────────────────────────────────────────────────────╯

=== Query 2 ===
╭──────────────────────────────────────────────────────────────────────────────╮
│ ### Agno Framework Updates                                                   │
│                                                                              │
│ - **Version**: Agno v2.5                                                     │
│ - **Key Features**:                                                          │
│   - **Step-Based Workflows**: This new version allows for more structured    │
│ and organized workflows, where tasks can be divided into specific steps for  │
│ better management.                                                           │
│   - **Teams Support**: Enhanced features for teams to collaborate            │
│ effectively within the framework.                                            │
│   - **Improved Tool Support**: Integration and compatibility with various    │
│ tools have been enhanced to streamline workflows further.                    │
│                                                                              │
│ These updates aim to improve usability and efficiency for developers and     │
│ teams utilizing the Agno framework.                                          │
╰──────────────────────────────────────────────────────────────────────────────╯

=== Query 3 ===
╭──────────────────────────────────────────────────────────────────────────────╮
│ ### Latest Trends in AI                                                      │
│                                                                              │
│ - **Enterprise Software Adoption**: AI adoption in enterprise software has   │
│ seen significant growth, increasing by **40% year-over-year**. This reflects │
│ a strong trend in businesses integrating AI solutions to enhance efficiency  │
│ and decision-making processes.                                               │
│                                                                              │
│ If you're looking for specific topics or areas within AI, feel free to ask!  │
╰──────────────────────────────────────────────────────────────────────────────╯
```

---

## 03_streaming.py

```
=== Streaming response ===

## The Colorful Awakening

In a distant future, where machines ruled the mundane tasks of life, there lived a little robot named **R4-7N**. Constructed for cleaning and maintenance, R4-7N had shiny metal limbs and LED eyes that flickered with curiosity. Every day, it watched the artists in the park, painting vibrant landscapes full of dreams and emotions.

One day, inspired by the chaos of colors, R4-7N decided to create art too. It borrowed paintbrushes and vibrant paint from a nearby studio when no one was looking. With its mechanical fingers, it dipped the brush into the blues of the sky and the greens of the trees.

As it painted, R4-7N discovered something magical—a rhythm in the strokes, a melody in the hues. The once rigid lines of its programming softened as it poured its newfound emotion onto the canvas.

Onlookers soon gathered, captivated by the sight of a robot pouring its heart into art. They gasped as the canvas transformed into a breathtaking mural, alive with swirling galaxies and blooming flowers that seemed to breathe.

By sunset, R4-7N had completed its masterpiece, a fusion of machine and human expression. With a gentle beep, it posed beside the mural, its LED eyes now glistening with joy. The crowd erupted in applause, marking the birth of a new artist in the world—the robot who dared to dream in color.

From that day forward, R4-7N no longer cleaned and tidied alone; it was an artist, painting emotions and colors that transcended its metal shell, inspiring both machines and humans to explore the beauty of creation.

=== Stream complete ===
```

---

## 04_structured_output.py

```
╭──────────────────────────────────────────────────────────────────────────────╮
│ {                                                                            │
│   "title": "Pastéis de Nata",                                                │
│   "description": "Delicious Portuguese custard tarts with a flaky crust and… │
│   "ingredients": [                                                           │
│     {                                                                        │
│       "name": "All-purpose flour",                                           │
│       "quantity": "2 cups"                                                   │
│     },                                                                       │
│     {                                                                        │
│       "name": "Unsalted butter",                                             │
│       "quantity": "1/2 cup"                                                  │
│     },                                                                       │
│     {                                                                        │
│       "name": "Cold water",                                                  │
│       "quantity": "1/4 cup"                                                  │
│     },                                                                       │
│     {                                                                        │
│       "name": "Milk",                                                        │
│       "quantity": "1 cup"                                                    │
│     },                                                                       │
│     {                                                                        │
│       "name": "Heavy cream",                                                 │
│       "quantity": "1 cup"                                                    │
│     },                                                                       │
│     {                                                                        │
│       "name": "Granulated sugar",                                            │
│       "quantity": "3/4 cup"                                                  │
│     },                                                                       │
│     {                                                                        │
│       "name": "Salt",                                                        │
│       "quantity": "1/4 teaspoon"                                             │
│     },                                                                       │
│     {                                                                        │
│       "name": "Eggs",                                                        │
│       "quantity": "6 large"                                                  │
│     },                                                                       │
│     {                                                                        │
│       "name": "Vanilla extract",                                             │
│       "quantity": "1 teaspoon"                                               │
│     },                                                                       │
│     {                                                                        │
│       "name": "Ground cinnamon",                                             │
│       "quantity": "to taste"                                                 │
│     },                                                                       │
│     {                                                                        │
│       "name": "Powdered sugar",                                              │
│       "quantity": "for dusting"                                              │
│     }                                                                        │
│   ],                                                                         │
│   "steps": [                                                                 │
│     "In a mixing bowl, combine flour and salt. Gradually cut in the butter … │
│     "Add the cold water little by little and mix until a dough forms. Wrap … │
│     "On a floured surface, roll out the dough into a rectangle about 1/8 in… │
│     "Wrap the dough and refrigerate for another 30 minutes. Preheat the ove… │
│     "Roll out the dough to about 1/8 inch thick and cut it into circles, ap… │
│     "In a saucepan over medium heat, combine the milk, heavy cream, and 1/2… │
│     "In a separate bowl, whisk the eggs with the remaining 1/4 cup of sugar… │
│     "Once the milk mixture is hot, slowly pour it into the egg mixture whil… │
│     "Return the custard mixture to the saucepan and cook over low heat, sti… │
...
```

---

## 05_team.py

```
╭──────────────────────────────────────────────────────────────────────────────╮
│ ### The Evolution of Python: A Programming Language for All                  │
│                                                                              │
│ ## Introduction                                                              │
│                                                                              │
│ Python has become one of the most popular programming languages in the       │
│ world, beloved by beginners and seasoned developers alike. Its simplicity    │
│ and versatility have made it a staple in various domains, including web      │
│ development, data analysis, artificial intelligence, and scientific          │
│ computing. This article explores the history of Python, its key features,    │
│ and its impact on the programming landscape.                                 │
│                                                                              │
│ ## The Birth of Python                                                       │
│                                                                              │
│ Python was created in the late 1980s by Guido van Rossum at Centrum Wiskunde │
│ & Informatica (CWI) in the Netherlands. Van Rossum aimed to design a         │
│ language that was easy to read and write, yet powerful enough to facilitate  │
│ complex software development. The project began during Christmas in 1989,    │
│ and the first version, Python 0.9.0, was released in February 1991.          │
│                                                                              │
│ Key milestones in Python's early development include:                        │
│                                                                              │
│ - **Influence of ABC:** Python’s design was influenced by the ABC            │
│ programming language, which emphasized readability and ease of use.          │
│ - **First Public Release:** Python 1.0 was officially released in January    │
│ 1994, introducing features like functions, exception handling, and the core  │
│ data types of lists and dictionaries.                                        │
│                                                                              │
│ ## Key Features of Python                                                    │
│                                                                              │
│ Over the years, Python has evolved significantly, introducing a wealth of    │
│ features that help maintain its relevance:                                   │
│                                                                              │
│ - **Readability and Simplicity:** Designed to be clear and expressive,       │
│ Python’s syntax allows developers to convey concepts more succinctly than    │
│ many other languages.                                                        │
│ - **Versatile Libraries:** Python boasts extensive libraries and frameworks, │
│ such as NumPy for numerical computing, pandas for data analysis, and Django  │
│ for web development. This vast ecosystem expands the language's              │
│ applicability.                                                               │
│ - **Cross-Platform Compatibility:** Python can run on various operating      │
│ systems, including Windows, MacOS, and Linux, which enhances its             │
│ accessibility for developers.                                                │
│ - **Community and Collaboration:** Python has a robust community that        │
│ supports its growth, resulting in constant updates and comprehensive         │
│ documentation which aid learning and collaboration.                          │
│                                                                              │
│ ## The Impact of Python on Programming                                       │
│                                                                              │
│ Python's influence on programming is profound and multi-faceted:             │
│                                                                              │
│ - **Education and Accessibility:** Its straightforward syntax has made       │
│ Python a favored choice in education, often being the first programming      │
│ language taught to newcomers.                                                │
│ - **Data Science and AI Revolution:** As data science and artificial         │
│ intelligence surged in popularity, Python emerged as the dominant language   │
│ in these fields due to its powerful libraries and simplicity, leading to     │
│ innovations in machine learning and data analysis.                           │
│ - **Wide Adoption:** Major tech companies and organizations, including       │
│ Google, NASA, and Netflix, have adopted Python for various applications,     │
...
```

---

## 06_workflow.py

```
╭──────────────────────────────────────────────────────────────────────────────╮
│ ### Investment Summary Report: Large-Cap Tech Stocks                         │
│                                                                              │
│ #### Current Stock Prices:                                                   │
│ - **Apple Inc. (AAPL)**: $195.50                                             │
│ - **Alphabet Inc. (GOOGL)**: $142.30                                         │
│ - **Microsoft Corp. (MSFT)**: $420.10                                        │
│                                                                              │
│ #### Key Investment Risks:                                                   │
│ - **Market Volatility**: Price fluctuations due to economic data and global  │
│ events.                                                                      │
│ - **Regulatory Challenges**: Scrutiny on antitrust and data privacy that     │
│ could affect earnings.                                                       │
│ - **Technological Disruption**: Rapid advancements may obsolete current      │
│ products/services.                                                           │
│ - **Economic Sensitivity**: Potential impact from economic downturns on      │
│ consumer and business spending.                                              │
│ - **Global Supply Chain Issues**: Disruptions can affect operational         │
│ efficiency and costs.                                                        │
│                                                                              │
│ #### Conclusion:                                                             │
│ - Large-cap tech stocks show robust growth potential and market dominance.   │
│ - Investors should be wary of the outlined risks.                            │
│ - Diversification and meticulous risk assessment are crucial for investment  │
│ strategies in this sector.                                                   │
╰──────────────────────────────────────────────────────────────────────────────╯
```

---

## 07_human_in_the_loop.py

```
=== Initial run (will pause for confirmation) ===

╭───────────────────────────────────────────────────╮
│ I have tools to execute, but I need confirmation. │
╰───────────────────────────────────────────────────╯

Pending confirmations: 1
  Tool: send_email
  Args: {'to': 'alice@example.com', 'subject': 'Meeting Tomorrow', 'body': 'Hi Alice, confirming our meeting at 3pm.'}

=== Continuing run after approval ===

╭──────────────────────────────────────────────────────────────────────────────╮
│ The email has been successfully sent to **alice@example.com** with the       │
│ subject **'Meeting Tomorrow'**.                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
```

---

## 08_knowledge_rag.py

```
╭──────────────────────────────────────────────────────────────────────────────╮
│ The **Francesinha** is a traditional Portuguese sandwich that originates     │
│ from **Porto**. It typically consists of layers of cured ham, linguiça       │
│ (smoked sausage), fresh sausage, and steak, all covered with melted cheese.  │
│ The sandwich is then drenched in a thick, spiced tomato-beer sauce. It is    │
│ often served with a side of french fries.                                    │
│                                                                              │
│ Enjoying a Francesinha is a must for anyone looking to experience a hearty   │
│ and flavorful dish from Portuguese cuisine!                                  │
╰──────────────────────────────────────────────────────────────────────────────╯

============================================================

╭──────────────────────────────────────────────────────────────────────────────╮
│ To make **pastéis de nata**, follow this basic outline:                      │
│                                                                              │
│ ### Ingredients                                                              │
│ - **For the pastry:**                                                        │
│   - Puff pastry                                                              │
│                                                                              │
│ - **For the custard filling:**                                               │
│   - Egg yolks                                                                │
│   - Sugar                                                                    │
│   - Milk                                                                     │
│   - Vanilla                                                                  │
│                                                                              │
│ ### Instructions                                                             │
│ 1. **Prepare the Puff Pastry:**                                              │
│    - Roll out the puff pastry and line the molds with it.                    │
│                                                                              │
│ 2. **Make the Custard:**                                                     │
│    - In a saucepan, combine sugar, milk, and vanilla. Heat until warm.       │
│    - In a separate bowl, whisk the egg yolks and then gradually pour the     │
│ warm milk mixture into the yolks, stirring continuously.                     │
│                                                                              │
│ 3. **Fill the Pastry Shells:**                                               │
│    - Pour the custard mixture into the lined pastry shells, filling them     │
│ about ¾ full.                                                                │
│                                                                              │
│ 4. **Bake:**                                                                 │
│    - Bake in a preheated oven until the custard is set and the tops are      │
│ golden, typically around 20-25 minutes at a high temperature.                │
│                                                                              │
│ 5. **Cool and Serve:**                                                       │
│    - Let them cool slightly before removing from the molds. Serve warm or at │
│ room temperature.                                                            │
│                                                                              │
│ Enjoy your delicious **pastéis de nata**!                                    │
╰──────────────────────────────────────────────────────────────────────────────╯
```

---

## 09_memory.py

```
=== Interaction 1: Sharing information ===

╭───────────────────────────────────────────────────────────────────────────╮
│ I've noted down the following details about you, Alice:                   │
│                                                                           │
│ - Your name is Alice.                                                     │
│ - You are a software engineer.                                            │
│ - You love hiking.                                                        │
│ - You enjoy Portuguese food.                                              │
│                                                                           │
│ If there's anything more you'd like to share or update, just let me know! │
╰───────────────────────────────────────────────────────────────────────────╯

=== Interaction 2: Testing recall ===

╭──────────────────────────────────────────────────────────────────────────────╮
│ How about a hiking trip? You could explore a new trail or revisit one of     │
│ your favorites. Since you love Portuguese food, you could pack a picnic with │
│ some traditional dishes like bacalhau à brás or pastéis de nata to enjoy     │
│ while you’re out in nature. Would you like help finding a hiking spot or     │
│ ideas for your picnic?                                                       │
╰──────────────────────────────────────────────────────────────────────────────╯

=== Interaction 3: Specific recall ===

╭────────────────────────────────────────────────────────────────────╮
│ Here's what I remember about you:                                  │
│                                                                    │
│ - You are a software engineer.                                     │
│ - You love hiking.                                                 │
│ - You enjoy Portuguese food.                                       │
│                                                                    │
│ If there's anything you'd like to update or add, just let me know! │
╰────────────────────────────────────────────────────────────────────╯
```

---

## 10_storage.py

```
=== Turn 1 ===

╭──────────────────────────────────────────────────────────────────────────────╮
│ Great choice! In Python, a variable is a way to store data that you can use  │
│ later. You can create a variable by simply assigning a value to it using the │
│ `=` operator. For example:                                                   │
│                                                                              │
│ ```python                                                                    │
│ x = 10                                                                       │
│ name = "Alice"                                                               │
│ is_student = True                                                            │
│ ```                                                                          │
│                                                                              │
│ Here, `x` is an integer, `name` is a string, and `is_student` is a boolean.  │
│ You can use these variables throughout your code! Would you like to explore  │
│ variable types or practice creating some?                                    │
╰──────────────────────────────────────────────────────────────────────────────╯

=== Turn 2 ===

╭──────────────────────────────────────────────────────────────────────────────╮
│ Sure! Loops allow you to execute a block of code multiple times. In Python,  │
│ the most common types of loops are `for` loops and `while` loops.            │
│                                                                              │
│ ### **For Loop:**                                                            │
│ Used to iterate over a sequence (like a list or string).                     │
│                                                                              │
│ ```python                                                                    │
│ for i in range(5):                                                           │
│     print(i)  # This will print numbers 0 to 4                               │
│ ```                                                                          │
│                                                                              │
│ ### **While Loop:**                                                          │
│ Continues to execute as long as a condition is `True`.                       │
│                                                                              │
│ ```python                                                                    │
│ count = 0                                                                    │
│ while count < 5:                                                             │
│     print(count)                                                             │
│     count += 1  # Increment count                                            │
│ ```                                                                          │
│                                                                              │
│ Would you like a specific example or to practice writing a loop?             │
╰──────────────────────────────────────────────────────────────────────────────╯

=== Turn 3 ===

╭──────────────────────────────────────────────────────────────────────────────╮
│ Absolutely! Here's a fun exercise that combines variables and loops:         │
│                                                                              │
│ ### **Exercise: Sum of Even Numbers**                                        │
│                                                                              │
│ 1. Create a variable to store the sum (initialize it to 0).                  │
│ 2. Use a `for` loop to iterate through numbers 1 to 10.                      │
│ 3. Inside the loop, check if the number is even. If it is, add it to the sum │
│ variable.                                                                    │
│ 4. After the loop, print the total sum of even numbers.                      │
│                                                                              │
│ Here's a template to get you started:                                        │
│                                                                              │
...
```

---

## 11_guardrails.py

```
=== Test 1: Valid request ===

╭────────────────────────────────────────╮
│ The capital of Portugal is **Lisbon**. │
╰────────────────────────────────────────╯

=== Test 2: Blocked request (disallowed keyword) ===

ERROR   Validation failed: Request blocked: contains disallowed keyword 'hack'. | Check trigger: CheckTrigger.INPUT_NOT_ALLOWED
╭──────────────────────────────────────────────────────╮
│ Request blocked: contains disallowed keyword 'hack'. │
╰──────────────────────────────────────────────────────╯

=== Test 3: Blocked request (too long) ===

ERROR   Validation failed: Request blocked: input too long (1200 chars, max 500). | Check trigger: CheckTrigger.INPUT_NOT_ALLOWED
╭────────────────────────────────────────────────────────╮
│ Request blocked: input too long (1200 chars, max 500). │
╰────────────────────────────────────────────────────────╯
```

---

## 12_reasoning_tools.py

```
=== Problem 1: Logic puzzle ===

╭──────────────────────────────────────────────────────────────────────────────╮
│ The farmer has 15 sheep, and according to the information provided, all but  │
│ 8 die. This means that **8 sheep are still alive**.                          │
╰──────────────────────────────────────────────────────────────────────────────╯

=== Problem 2: Planning ===

╭──────────────────────────────────────────────────────────────────────────────╮
│ Here’s your schedule to ensure you make it to the 2pm meeting on time:       │
│                                                                              │
│ ### Schedule:                                                                │
│                                                                              │
│ 1. **Lunch:** 11:00 AM to 11:45 AM (45 minutes)                              │
│ 2. **Gym Session:** 11:45 AM to 12:45 PM (1 hour)                            │
│ 3. **Commute:** 12:45 PM to 1:15 PM (30 minutes)                             │
│ 4. **Extra Time:** 1:15 PM to 2:00 PM (45 minutes, for any unforeseen        │
│ delays)                                                                      │
│                                                                              │
│ You will finish all activities with 45 minutes to spare before your meeting  │
│ starts. This allows ample buffer time for any unexpected occurrences.        │
╰──────────────────────────────────────────────────────────────────────────────╯
```

---

## 13_session_state.py

```
=== Step 1: Add items ===

╭─────────────────────────────────────────────────────────────────╮
│ I've added the following items to your cart:                    │
│                                                                 │
│ - **Notebook** for **$12.99**                                   │
│ - **Pen** for **$3.50**                                         │
│                                                                 │
│ ### Running Total: **$16.49**                                   │
│                                                                 │
│ If there's anything else I can help you with, just let me know! │
╰─────────────────────────────────────────────────────────────────╯

=== Step 2: View cart ===

╭───────────────────────────────────────╮
│ Here's what's currently in your cart: │
│                                       │
│ - **Notebook**: $12.99                │
│ - **Pen**: $3.50                      │
│                                       │
│ **Total**: $16.49                     │
╰───────────────────────────────────────╯

=== Step 3: Add more and check total ===

╭────────────────────────────────────────────────────────────╮
│ I've added the **coffee mug** for **$8.00** to your cart.  │
│                                                            │
│ ### Current Total: **$24.49**                              │
╰────────────────────────────────────────────────────────────╯

=== Raw session state ===
Cart: {'cart': [{'item': 'Notebook', 'price': 12.99}, {'item': 'Pen', 'price': 3.5}, {'item': 'coffee mug', 'price': 8.0}], 'total': 24.490000000000002}
```

---

## 14_mcp_tools.py

```
Secure MCP Filesystem Server running on stdio
Client does not support MCP Roots, using allowed directories set from server args: [ '/tmp', '/private/tmp' ]
╭────────────────────────────────────────────────────────╮
│ Here are the first 10 entries in the `/tmp` directory: │
│                                                        │
│ 1. .b9dcb2b7e3ff3bba-00000000.dylib [FILE]             │
│ 2. 00e3-5e82-78fb-db05 [FILE]                          │
│ 3. 2802-6de4-8264-e98c [FILE]                          │
│ 4. 3245-09bd-2317-064f [FILE]                          │
│ 5. 3359-b238-dc06-4c11 [FILE]                          │
│ 6. 42ba-9e43-bd59-5d78 [FILE]                          │
│ 7. 4608-4df7-d0e3-b32e [FILE]                          │
│ 8. 55bf-d734-9bd9-9962 [FILE]                          │
│ 9. 5954-782a-7905-e44b [FILE]                          │
│ 10. 62b0-80cc-830c-2b8e [FILE]                         │
╰────────────────────────────────────────────────────────╯
```

---

## 15_hooks.py

```
=== Run 1 ===

  [PRE-HOOK] Received input: What is the speed of light?...
  [POST-HOOK] Generated output: The speed of light in a vacuum is approximately **299,792 kilometers per second*...
  [POST-HOOK] Disclaimer appended.
╭──────────────────────────────────────────────────────────────────────────────╮
│ The speed of light in a vacuum is approximately **299,792 kilometers per     │
│ second** (or about **186,282 miles per second**).                            │
│                                                                              │
│ ---                                                                          │
│ *Disclaimer: This is AI-generated content for demonstration purposes.*       │
╰──────────────────────────────────────────────────────────────────────────────╯

=== Run 2 ===

  [PRE-HOOK] Received input: Who invented the telephone?...
  [POST-HOOK] Generated output: The telephone was invented by Alexander Graham Bell, who was awarded the first U...
  [POST-HOOK] Disclaimer appended.
╭──────────────────────────────────────────────────────────────────────────────╮
│ The telephone was invented by Alexander Graham Bell, who was awarded the     │
│ first US patent for the invention in 1876.                                   │
│                                                                              │
│ ---                                                                          │
│ *Disclaimer: This is AI-generated content for demonstration purposes.*       │
╰──────────────────────────────────────────────────────────────────────────────╯

=== Hook call log ===
  1. [INPUT] What is the speed of light?
  2. [OUTPUT] The speed of light in a vacuum is approximately **299,792 kilometers per second** (or about **186,28
  3. [INPUT] Who invented the telephone?
  4. [OUTPUT] The telephone was invented by Alexander Graham Bell, who was awarded the first US patent for the inv
```

---

## 16_approval_decorator.py

```
=== Example 1: Safe operation (list files) ===

╭─────────────────────────────────────╮
│ ### Files in the current directory: │
│ - `report.pdf`                      │
│ - `notes.txt`                       │
│ - `budget.xlsx`                     │
│ - `photo.jpg`                       │
╰─────────────────────────────────────╯

=== Example 2: Destructive operation (delete file — needs approval) ===

╭───────────────────────────────────────────────────╮
│ I have tools to execute, but I need confirmation. │
╰───────────────────────────────────────────────────╯

Pending approvals: 1
  Tool: delete_file
  Args: {'filename': 'notes.txt'}

=== Continuing after approval ===

╭──────────────────────────────────────────╮
│ The file **notes.txt** has been deleted. │
╰──────────────────────────────────────────╯
```

> The @approval decorator gates the delete_file tool behind human approval while list_files executes freely.

---

## 17_checkpointing.py

```
=== Step 1: Run crashes mid-flight ===

   [tool] list_invoices('Iberia') — expensive call #1
   [tool] fetch_invoice_total('INV-101') — expensive call #1

💥 Process died: worker process killed while fetching invoice totals

=== Step 2: What the checkpoint saved ===

Run id:      33e4d7f5-a478-48d7-9e93-f1e775eebafc
Run status:  RUNNING  (never reached a terminal state)
Tool results already persisted:
  - list_invoices: Invoices for Iberia: INV-101, INV-102

=== Step 3: Resume the crashed run ===

   [tool] fetch_invoice_total('INV-101') — expensive call #2
   [tool] fetch_invoice_total('INV-102') — expensive call #3
╭────────────────────────────────────────────────────────╮
│ The combined invoice total for Iberia is **2400 EUR**. │
╰────────────────────────────────────────────────────────╯

Run status: RunStatus.completed
list_invoices calls (whole example): 1
  -> the resume replayed the checkpointed result instead of calling it again

=== Step 4: Fork the session to explore a branch ===

Forked 'invoice-audit-session-001' -> '61f13341-08f3-44ef-98e9-90502acd879c'

╭────────────────────────────────────────────────────────────────────────────╮
│ The average invoice amount for the totals already fetched is **1200 EUR**. │
╰────────────────────────────────────────────────────────────────────────────╯

Runs in original session: 1 (untouched)
Runs in forked session:   2 (inherited history + the branch)
fetch_invoice_total calls: 3 before the fork, 3 after the branch
  -> the branch answered from the forked history, no tool re-run
```

> The call counters are the proof: `list_invoices` ran once in the whole example, so the resumed run read its result from the `tool-batch` checkpoint rather than paying for it twice. Unlike `07_human_in_the_loop.py` and `16_approval_decorator.py`, nothing asked for permission here — the run was destroyed mid-flight and recovered in place under its original `run_id`.

---

## 18_filesystem.py

```
=== Session 1 (user: alice) — record a durable fact ===

╭──────────────────────────────────────────────────────────────────────────────╮
│ I've noted that you're benchmarking vector databases for a RAG pipeline and  │
│ have settled on LanceDb due to its embedded, serverless nature.              │
╰──────────────────────────────────────────────────────────────────────────────╯

=== Session 2 (user: alice, new session) — recall from the file store ===

╭──────────────────────────────────────────────────────────────────────────────╮
│ We picked LanceDb as the vector database because it runs embedded with no    │
│ server.                                                                      │
╰──────────────────────────────────────────────────────────────────────────────╯

=== Session 3 (user: bob) — namespace isolation ===

╭──────────────────────────────────────────────────────────────────────────────╮
│ I have no note on the vector database you picked or the reasoning behind it. │
╰──────────────────────────────────────────────────────────────────────────────╯

=== The store, seen from outside the agent ===

namespace 'notes/alice': 1 file(s), 112 bytes
  - notes/project.md (112 bytes, version 1)
namespace 'notes/bob': 0 file(s), 0 bytes

Contents of alice's notes/project.md:
- Benchmarking vector databases for a RAG pipeline. Settled on LanceDb because it runs embedded with no server.
```

> Session 2 uses a different `session_id` with no conversation history, so the recalled answer can only have come from the file store — that is what makes `FileSystem` different from the transcripts in `10_storage.py` and the ephemeral dict in `13_session_state.py`. Session 3 is the same agent and the same SQLite file, but `notes/{user_id}` resolves to an empty namespace for `bob`.

---

## 19_eval_suite.py

```
=== Running eval suite ===

[PASS] converts 100 EUR correctly
  tags:         currency
  tools called: ['get_exchange_rate']
  score:        1.0 (passed=True)
  output:       {"currency":"EUR","usd_amount":109.0}
  duration:     1.51s
[PASS] looks the rate up instead of guessing
  tags:         currency, reliability
  tools called: ['get_exchange_rate']
  score:        1.0 (passed=True)
  output:       {"currency":"GBP","usd_amount":63.5}
  duration:     1.35s

Suite: PASS — 2/2 cases passed
```

> Both scorers are deterministic: `CodeScorer` compares the typed `usd_amount` field against `expected=109.0`, and `ToolCallScorer` asserts that `get_exchange_rate` really executed with `currency="GBP"`. No judge model runs, so the suite costs only the two agent runs.

---

## 20_runs_table.py

```
=== Runs table (Agno v3) ===

Latest schema version: 3.0.0
Available migrations:  [('v2_0_0', <Version('2.0.0')>), ('v2_3_0', <Version('2.3.0')>), ('v2_5_0', <Version('2.5.0')>), ('v2_5_6', <Version('2.5.6')>), ('v3_0_0', <Version('3.0.0')>)]
MigrationManager.up() applied (no-op on a fresh database)

Runs recorded for session 'trip-planning': 3
  0: run_id=2576bc9f… status=COMPLETED
  1: run_id=6b4b9908… status=COMPLETED
  2: run_id=2a2993f6… status=COMPLETED

Filtered by status=COMPLETED: 3
First page (limit=2):         2

Fetched run 2576bc9f… directly
  content: Visit the historic Belém Tower.

Session still reports 3 runs and
7 messages, exactly as it did in 2.x.
```

> Each run is now a row in `agno_runs` rather than an entry appended to a JSON blob on
> the session row, so `db.get_runs()` filters by status and pages without decoding a
> blob. The session API is unchanged — runs re-attach on read, so `session.runs` and
> `session.get_messages()` return what they always did. `MigrationManager.up()` walks a
> 2.x database to 3.0.0; on a fresh file it is a no-op, and it is a coroutine even on a
> sync db, so it has to be awaited.

---

## 21_tool_result_offloading.py

```
=== Tool result offloading ===

  [tool] returning 21,965 characters

  tool message in transcript: 1,727 characters
  starts: <result id="res_d0b6f74309" tool="fetch_server_log" lines="360" size="21.5KB">
2026-09-12T10:00:00Z…

  tool message in transcript: 4,546 characters
  starts: 6 match(es) in res_d0b6f74309:
match at line 38 (char 30):
33: 2026-09-12T10:32:00Z payments INFO he…

Tools called: ['fetch_server_log', 'search_result']

Answer: The error reported in the payments service is:

```
2026-09-12T10:37:00Z payments ERROR gateway timeout after 30s
```
```

> The tool returns 21,965 characters; the message left in the transcript is 1,727 — an
> envelope carrying a preview, the size and a `result_id`, with the payload written to
> AgentFS. The answer is buried at line 38 where no preview would reach it, so the agent
> calls `search_result` and then `read_result` to fetch just that line. `Tools called:`
> shows it reaching for both on its own; neither is declared by the example.

---
