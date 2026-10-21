---
Data: 2026-10-11T23:09:00
Tags:
  - note
  - youngling
Connection:
  - "[[Competitive Programming and Contests]]"
Area: "[[Master's Degree.base]]"
---
# Job Sequencing

**Problem**: Given two arrays, `deadline[]` and `profit[]`, where `deadline[i]` is the last time unit by which the i-th job must be completed, and `profit[i]` is the profit earned from completing it. Each job takes 1 unit time, and only one job can be scheduled at a time. A job earns profit only if finished within its deadline. **Find the number of jobs completed and maximum profit**.

**Examples**:
- **Input:** `deadline[] = [4, 1, 1, 1]`, `profit[] = [20, 10, 40, 30]`  
- **Output:** `[2, 60]`  
- **Explanation:** Job 1 (profit 20, deadline 4) can be scheduled. Among the three jobs with deadline 1, only one fits, so e pick the highest profit (40). Hence, 2 jobs with total profit = 60.

- **Input:** `deadline[] = [2, 1, 2, 1, 1]`, `profit[] = [100, 19, 27, 25, 15]`  
- **Output:** `[2, 127]`  
- **Explanation:** Picking the job with profit 100 (deadline 2) and the job with profit 27 (deadline 2); they can occupy the two available slots before deadline 2. Thus 2 jobs are scheduled for a maximum total profit of 127.

The idea is to sort the jobs based **on their deadlines in ascending order**. This ensures that jobs with **earlier deadlines are processed first**, preventing situations where a job with a short deadline remains unscheduled because a job with a later deadline was chosen instead. We use a **min-heap to keep track of the selected job**s, allowing us to efficiently replace lower-profit jobs when a more profitable job becomes available.

```c++
vector<int> jobSequencing(vector<int> &deadline, vector<int> &profit) {
    
    int n = deadline.size();
    vector<int> ans = {0, 0};
    vector<pair<int, int>> jobs;
    for (int i = 0; i < n; i++) {
        jobs.push_back({deadline[i], profit[i]});
    }

    // sort the jobs based on deadline
    // in ascending order
    sort(jobs.begin(), jobs.end());
    priority_queue<int, vector<int>, greater<int>> pq;

    for (int i = 0; i < jobs.size(); i++) {
        
        // if job can be scheduled within its deadline
        if (jobs[i].first > pq.size())
            pq.push(jobs[i].second);
        
        // replace the job with the lowest profit
        else if (!pq.empty() && pq.top() < jobs[i].second) {
            pq.pop();
            pq.push(jobs[i].second);
        }
    }

    while (!pq.empty()) {
        ans[1] += pq.top();
        pq.pop();
        ans[0]++;
    }

    return ans;
}
```

**Step by Step implementation:**
1. Store jobs as pairs of (Deadline, Profit).
2. Sort Jobs array in ascending order of deadline.
3. For each job in the sorted list:
4. If the job can be scheduled within its deadline, push its profit into the heap.
5. If the heap is full (equal to deadline), replace the existing lowest profit job with the current job if it has a higher profit.
6. This ensures that we always keep the most profitable jobs within the available slots.
7. Traverse through the heap and store the total profit and the count of jobs.

With this algorithm, using a **min-heap**, we have a **time complexity** of $O(n\log n)$ and a **space complexity** of $O(n)$
# References
- https://www.geeksforgeeks.org/dsa/job-sequencing-problem/