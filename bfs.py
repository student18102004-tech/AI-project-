from collections import deque


CAREER_GRAPH = {

    "Computer Science": [
        "Software Development",
        "Data Science",
        "Cybersecurity",
        "AI/ML"
    ],

    "Software Development": [
        "Web Development",
        "App Development",
        "Backend Development"
    ],

    "Web Development": [
        "Frontend Development",
        "Backend Development",
        "Full Stack Development"
    ],

    "Data Science": [
        "Data Analysis",
        "Machine Learning",
        "Business Intelligence"
    ],

    "AI/ML": [
        "Deep Learning",
        "NLP",
        "Computer Vision"
    ],

    "Cybersecurity": [
        "Network Security",
        "Ethical Hacking"
    ]
}



def bfs_career_exploration(start_category):

    visited = []
    queue = deque([start_category])

    while queue:

        node = queue.popleft()

        if node not in visited:

            visited.append(node)

            for neighbor in CAREER_GRAPH.get(node, []):

                if neighbor not in visited and neighbor not in queue:
                    queue.append(neighbor)

    return visited