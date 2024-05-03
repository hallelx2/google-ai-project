#!/usr/bin/env python
from business_analyst.crew import BusinessAnalystCrew


def run():
    # Replace with your inputs, it will automatically interpolate any tasks and agents information
    inputs = {
        'topic': 'AI LLMs'
    }
    BusinessAnalystCrew().crew().kickoff(inputs=inputs)