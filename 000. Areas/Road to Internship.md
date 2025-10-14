---
tags:
  - "#area"
---
# Road to Internship
## List of Company

#### General Big Tech
- [X] [Google](https://www.google.com/about/careers/applications/jobs/results/?company=Fitbit&company=Google&company=YouTube&distance=50&employment_type=INTERN)
- [x] [Meta](https://www.metacareers.com/jobs?teams[0]=University%20Grad%20-%20Business&teams[1]=University%20Grad%20-%20Engineering%2C%20Tech%20%26%20Design&teams[2]=University%20Grad%20-%20PhD%20%26%20Postdoc)
- [x] [Amazon](https://www.amazon.jobs/it/teams/internships-for-students)
- [X] [AMD](https://careers.amd.com/students/jobs?keywords=intern&categories=Student%20%2F%20Intern%20%2F%20Temp&page=1)
- [X] [Nvidia](https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite?workerSubType=0c40f6bd1d8f10adf6dae42e46d44a17&workerSubType=ab40a98049581037a3ada55b087049b7)
- [ ] [Spotify](https://www.lifeatspotify.com/students)
- [ ] [Netflix](https://explore.jobs.netflix.net/careers)
- [X] [Apple](https://jobs.apple.com/en-us/search?team=internships-STDNT-INTRN)
- [X] [Microsoft](https://jobs.careers.microsoft.com/global/en/search?q=intern&p=Software%20Engineering&exp=Students%20and%20graduates&l=en_us&pg=1&pgSz=20&o=Relevance&flt=true)
- [X] [Intel](https://intel.wd1.myworkdayjobs.com/en-US/External/details/Compiler-Development-Intern_JR0276991?q=software&workerSubType=dc8bf79476611087dfde99931439ae75&jobFamilyGroup=dc8bf79476611087d67b36517cf17036)
- [X] [Stripe](https://stripe.com/jobs/listing/software-engineer-intern/7206494)
- [x] [Cloudflare](https://www.cloudflare.com/it-it/careers/jobs/?department=Early+Talent)
- [X] [DataBricks]()
- [x] [IBM]()
- [x] [Revolut](https://www.revolut.com/careers/?text=rev-celerator+internship)
- [x] [ARM](https://earlycareers-arm.icims.com/jobs/15805/software-engineering-intern/job?mode=submit_apply) 
#### Computer Graphics Big Tech
- [X] [Activition](https://careers.activision.com/search-results?keywords=intern)
- [X] [Adobe](https://careers.adobe.com/us/en/search-results?qkexperienceLevel=University%20Intern)
- [ ] [Blizzard](https://careers.blizzard.com/global/en/c/internships-jobs)
- [X] [EA](https://jobs.ea.com/en_US/careers/Home/?4536=%5B8301%5D&4536_format=3019&4537=%5B8693%5D&4537_format=3020&listFilterMode=1&jobRecordsPerPage=20&)
- [X] [Epic Games](https://www.epicgames.com/site/en-US/careers/jobs?type=Intern&department=Engineering&page=1)
- [ ] [Nintendo](https://careers.nintendo.com/job-openings/?search=internship)
- [ ] [Sony](https://www.sonyjobs.com/jobs.html)
- [ ] [Ubisoft](https://www.ubisoft.com/en-us/company/careers/search?query=internship)
- [ ] [Unity](https://unity.com/careers/positions?title=intern)
- [X] [Larian Studio]()
- [X] [Autodesk](https://autodesk.wd1.myworkdayjobs.com/en-US/uni/details/Intern--Software-Engineer_25WD91327-1?q=itern)

## Coding Interview Topics

### Algorithms
- [Neetcode Roadmap](https://neetcode.io/roadmap)
- [Leetcode](https://leetcode.com/explore/learn)
### Languages
- [CPP Interview questions](https://www.interviewbit.com/cpp-interview-questions/)
- [General C++ Questions](https://hellointern.in/blog/c-interview-questions-and-answers-for-internship-88141)
- [Reddit Questions](https://www.reddit.com/r/cpp/comments/17r95li/questions_from_one_job_interview_on_c_developer/)
### Computer Graphics
- [Common Computer Graphics Questions](https://erkaman.github.io/posts/junior_graphics_programmer_interview.html)
- [Reddit Questions](https://www.reddit.com/r/GraphicsProgramming/comments/194ewll/graphics_programming_interview_prep/)

## Behavioral Interview
- [STAR-L SuperHero Valley](https://wiki-superherovalley-fun.translate.goog/preparation/behavioral/?_x_tr_sl=auto&_x_tr_tl=en&_x_tr_hl=en)
### Questions
##### Must to know
###### So, tell me a bit about yourself
**A:** I'm a Master's Computer Science student at the University of Pisa in Italy. My bachelor's degree was also in Computer Science at Pisa. During my studies, I had a work experience in a new startup in Milan that is developing an app for financial education and investments. In this experience, I worked on the backend using Python and PostgreSQL and implemented key services like authentication, payment systems, and newsletters. I also engineered the cloud infrastructure for all services to reduce costs and promote future scalability.

I also collaborated with a research team in Pisa focused on computer graphics and 3D modeling. I contributed to a couple of large open-source projects developed by them using C++, and I developed a modern system for line and polyline rendering for visual data analysis. At the moment, I still collaborate with them on many projects, such as the analysis of datasets for machine learning models in the architectural field. 

Outside of work and school, I'm working on many personal projects, like a system for distributed mesh simplification. In general, I spend much of my time on computer science and nerd stuff, but not only. I love sports and the gym, and I practice almost every day. At the moment, I’m searching for an internship in some computer science topics to improve my skills and expand my knowledge.


###### Why do you want to work here?
**A:** I want to work at Databricks because it's a company leading the way in data and AI innovation, and I'm excited about contributing to solutions that have such a wide global impact.

What motivates me most is the change to help customers unlock the full value of their data, whether that's though optimizing pipelines, enabling machine learning, or making analytics more accessible across a business. Databricks collaborative culture and its mission to make big data simple and actionable really align with my own career values.

With my studies and experiences in cloud platforms and software engineering for high performance I believe I can add real value by solving complex problems and helping customers achieve measurable business outcomes with Databricks technology.

###### Tell me about a time when you worked on a project with limited direction. What did you do?

**ANSWER 1**
**Situations**: During one of my master degree course I was asked to implemented an efficiente algorithm for mesh processing using gpgpu techniques, but there was little clarity on what the end result should be.
**Task**: My responsibility was to identify the key issues on the naive solution proposed and find a way to improve the efficiency to reduce the time and memory used by it.
**Action**: I clarified the pain points with profiling tools and I found that many of the time was spent on data reading from global memory. I then engineered a new structure of algorithms that allow to use in a more efficiency way the data to reduce the memory access storing information in shared memory. From this I taking the initiative to redesigned how the algorithm process information and how store it.
**Result**: As a result, the time to process the same amount of data decrease up to 30%, which significantly improved how it can be used in real situations
**Learning**: From this experience, I learned the importance of asking the right questions, taking initiative and using technical expertise to bring clarity where direction is limited. These are skills I would bring to projects here at Databricks.

**ANSWER 2**
**Situations:** During my experience at a startup building a platform for financial education and investments, I was asked with designing a cloud architecture to lower costs compared to the existing solution. 
**Task:** My responsibility was to identify the services that contributed most to spending and define alternatives that would reduce costs while preserving maximum flexibility for future scalability.
**Action**: First, I clarified the pain points using profiling and cost-analysis tools, finding that a significant portion of costs came from database services. I also discovered that, with the current configuration, the cost of backend machines would quickly become unsustainable with scaling. I then redesigned the cloud architecture by simplifying it and adopting lower-cost services. After a series of tests, I selected more appropriate instance sizes for the backend and configured autoscaling with a load balancer to handle peak traffic. I also optimized database settings and provisioning policies to align with real usage patterns.
**Result**: The outcome was an infrastructure that remained reliable and efficient while reducing costs by up to 50% and improving elastic scalability
**Learning**: From this experience, I learned the importance of asking the right questions, taking initiative, and using technical expertise to bring clarity when direction is limited. These are skills I would bring to projects at Databricks.

###### Describe a moment when you stayed true to your values, even when it was hard
**Situations**: During my Bachelor's thesis, I was asked to quickly run benchmarks on my implementation to meet a tight deadline with my supervisor.
**Task**: My task was to build a complete benchmark suite to measure how one of my implementations performed under stress in terms of time and memory, but I noticed that the requested parameters were not detailed enough to analyze the implementation properly and could lead to misleading results.
**Action**: Although there was pressure to push it through, I stayed true to my values of accuracy and transparency. I flagged the issue with my supervisor, explained why the parameters were insufficient to analyze performance, and proposed a revised benchmark for this and future implementations that would extract more information from tests and combine it into a complete analysis.
**Result**: This delayed delivery by 3–4 days, but the final benckmarks was complete, accurate, and clearly showed the strengths and weaknesses of my work
**Learning**: From this experience, I learned that standing by your values, even under pressure, builds long-term trust and credibility. At Databricks, where data analysis is fundamental, I would always prioritize doing the right thing for customers and the business.

###### What is DataBricks, and what are its key features?
**A:** Databricks is a unified analytics platform that combines data engineering, data science, and machine learning. It's key features include an interactive workspace for collaboration, support for multiple languages like python, scala and sql, as well as integrated data management solutions.  Databricks streamlines the process of building and managing data pipelines with delta lake, which provides ACID transaction and salable metadata handling. The platform also offers built-in machine learning tools, real-time stream processing, and seamless integration with various data sources and cloud services enhancing productivity for data teams.

###### Where do you see yourself in 5 years?
**A:** In five years, I see myself as a strong software engineer with advanced skills in optimization, problem solving, and managing complex systems. I’d like to be in a stimulating environment, such as Databricks, where I can leverage my abilities to improve services used by thousands of users and, above all, continue growing both technically and as a leader. I envision being able to manage contexts where others depend on me and to guide them toward achieving better outcomes for the company and, most importantly, for customers.
###### What do you do outside of work?
**A:** Outside school I work on many personal projects, one of the least is a system to do mesh simplification in a parallel and distributed way. In general computer science it's not only my school subject but also a passion and in my free time I like to read or watch video about many topics ralated with computer science. I also love sport and I practice it almost every days. 
###### What are your strengths and weaknesses?
**A:** One of my biggest strengths is the ability to organize my time and handle difficult situations with analytical and adaptive skills. I also have good leadership abilities; in fact, in many school contexts I have felt very comfortable managing large team projects and leading my teammates to achieve the goal with the best results possible. In the other hand, while I’m comfortable with one-on-one negotiations or small-group settings, I would say that presenting complex topics to large audiences, especially not in my native language, it is has challenging for me, and public speaking is a weakness that I recognize. To improve it, I have taken steps such as putting myself to the test in contexts where I need to present not-too-complex topics to multiple people in a non-native language, in order to challenge myself. I also practice with people I know to prepare myself before a talk. These efforts have significantly improved my confidence and my ability to communicate effectively, but it is still a work in progress.
###### Tell me about a project/class that you enjoyed/worked on.
**Situation:** One of the project-based classes that engaged me the most and that I loved developing, was High Performance Computing, which I took last year. In this course, we studied the fundamentals of parallel programming and the main patterns used in streaming and data-flow contexts. We also delved into GPU architectures, their execution model and memory hierarchy, working with real-world code examples in C++ and CUDA.
**Task:** The final project required implementing an algorithm of our choice efficiently in CUDA and optimizing it over the CPU and parallel versions by leveraging all the knowledge we had acquired
**Action**: Throughout the project, I applied these concepts by first conducting an in-depth study of the algorithm’s behavior and how it could be mapped to a parallel GPU context; then I built an initial implementation and ran a series of tests and profiling sessions to identify bottlenecks
**Result** I iterated on this approach until I achieved a fully working result with performance significantly above the baseline
**Learning**: This experience allowed me to explore an area I had never studied before and to test my skills in a real project; moreover, I was able to apply an analytical approach to the problem, analyzing the algorithm and progressively identifying bottlenecks to optimize, a skill I intend to use to improve the services offered by Databricks.
##### Others
## Resources 
- [Cracking the Code Interview](https://github.com/AatmikJain/ComputerScienceBooks/blob/master/Cracking%20the%20Coding%20Interview.pdf)
- [SuperHero Valley Wiki](https://wiki.superherovalley.fun/preparation/intro/)