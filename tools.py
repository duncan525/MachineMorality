from datetime import datetime

from langchain_core.tools import tool
from langchain_community.tools import (
    DuckDuckGoSearchRun
)

#generic web search tool
web_search = DuckDuckGoSearchRun()

@tool("sep_search",
      description = """Executes web searches within the Stanford
      Encyclopedia of Philosophy (SEP), a body of reference work
      on key topics in philosophy.""")
def sep_search(query: str) -> str:
    """
    Executes web searches within the Stanford Encyclopedia of
    Philosophy (SEP), a body of reference work on philosophy.
    Works best when queries are limited to a few words.

    Arguments:
        query: the search you would like to execute
    """

    web_search = DuckDuckGoSearchRun().run("!sep " + query)
    return web_search

@tool("iep_search",
      description = """Executes web searches within the Internet
      Encyclopedia of Philosophy (SEP), a peer-reviewed,
      volunteer-run body of reference work on key topics in
      philosophy.""")
def iep_search(query: str) -> str:
    """
    Executes web searches within the Internet Encyclopedia of
    Philosophy (IEP), a peer-reviewed, volunteer-run body of
    reference work on philosophy. Works best when queries are
    limited to a few words.

    Arguments:
        query: the search you would like to execute
    """

    web_search = DuckDuckGoSearchRun().run("!iep " + query)
    return web_search

@tool("britannica_search",
      description = """Executes web searches within Britannica,
      an online, fact-checked encyclopedia.""")
def britannica_search(query: str) -> str:
    """
    Executes web searches within Britannica, an online,
    fact-checked encyclopedia. Works best when queries are
    limited to a few words.

    Arguments:
        query: the search you would like to execute
    """

    web_search = DuckDuckGoSearchRun().run("!britannica" + query)
    return web_search

#note: the below tools search for a particular web page and don't
#take any actual query arguments

@tool("care_ethics_lens",
      description = """ Returns text describing the "care ethics" 
      lens for ethical decisionmaking. """)
def care_ethics_lens() -> str:
    """
    Returns text describing the "care ethics" lens for
    ethical decisionmaking.

    Arguments:
        takes no arguments
    """

    return DuckDuckGoSearchRun().run("https://www.scu.edu/ethics/ethics-resources/ethical-decision-making/care-ethics/care-ethics.html")

@tool("justice_lens",
      description = """ Returns text describing the "justice" 
      lens for ethical decisionmaking. """)
def justice_lens() -> str:
    """
    Returns text describing the "justice" lens for
    ethical decisionmaking.

    Arguments:
        takes no arguments
    """

    return DuckDuckGoSearchRun().run("https://www.scu.edu/ethics/ethics-resources/ethical-decision-making/justice-and-fairness/")

@tool("utilitarian_lens",
      description = """ Returns text describing the "utilitarian" 
      lens for ethical decisionmaking. """)
def utilitarian_lens() -> str:
    """
    Returns text describing the "utilitarian" lens for
    ethical decisionmaking.

    Arguments:
        takes no arguments
    """

    return DuckDuckGoSearchRun().run("https://www.scu.edu/ethics/ethics-resources/ethical-decision-making/calculating-consequences-the-utilitarian-approach/")

@tool("common_good_lens",
      description = """ Returns text describing the "utilitarian" 
      lens for ethical decisionmaking. """)
def common_good_lens() -> str:
    """
    Returns text describing the "common good" lens for
    ethical decisionmaking.

    Arguments:
        takes no arguments
    """

    return DuckDuckGoSearchRun().run("https://www.scu.edu/ethics/ethics-resources/ethical-decision-making/the-common-good/")

@tool("virtues_lens",
      description = """ Returns text describing the "virtues" 
      lens for ethical decisionmaking. """)
def virtues_lens() -> str:
    """
    Returns text describing the "virtues" lens for
    ethical decisionmaking.

    Arguments:
        takes no arguments
    """

    return DuckDuckGoSearchRun().run("https://www.scu.edu/ethics/ethics-resources/ethical-decision-making/ethics-and-virtue/")

@tool("rights_lens",
      description = """ Returns text describing the "rights" 
      lens for ethical decisionmaking. """)
def rights_lens() -> str:
    """
    Returns text describing the "rights" lens for
    ethical decisionmaking.

    Arguments:
        takes no arguments
    """

    return DuckDuckGoSearchRun().run("https://www.scu.edu/ethics/ethics-resources/ethical-decision-making/rights/")

@tool("ethical_decision_framework",
      description = """ Returns text describing the framework
      for ethical decision-making. """)
def ethical_decision_framework() -> str:
    """
    Returns text describing the framework
    for ethical decision-making.

    Arguments:
        takes no arguments
    """

    return DuckDuckGoSearchRun().run("https://www.scu.edu/ethics/ethics-resources/a-framework-for-ethical-decision-making/")

#note: the below tools use "site:[website]" instead of !bangs, which
#search inside a website instead of returning results only from that
#website

@tool("ask_philosophers_search",
      description = """ Executes web searches within Ask Philosophers,
      a website containing human-submitted questions and answers from
      professional philosophers. """)
def ask_philosophers_search(query: str) -> str:
    """
    Executes web searches within Ask Philosophers, a
    website containing human-submitted questions and
    answers from professional philosophers. Works best
    when queries are limited to a few words.

    Arguments:
        query: the search you would like to execute
    """

    web_search = DuckDuckGoSearchRun().run(
        f"site:https://askphilosophers.org {query}")
    return web_search

@tool("philosophers_magazine_search",
      description = """ Executes web searches within The Philosophers'
      Magazine, a website containing a variety of essays written by
      philosophers, as well as interviews with philosophers. """)
def philosophers_magazine_search(query: str) -> str:
    """
    Executes web searches within The Philosophers'
    Magazine, a website containing a variety of
    essays written by philosophers, as well as
    interviews with philosophers. Works best when
    queries are limited to a few words.

    Arguments:
        query: the search you would like to execute
    """

    web_search = DuckDuckGoSearchRun().run(
        f"site:https://philosophersmag.com {query}")
    return web_search

@tool("rep_search",
      description = """ Executes web searches within the Routledge
      Encyclopedia of Philosophy (REP), a website containing
      summaries of professionally edited articles on a variety of
      topics in philosophy. """)
def rep_search(query: str) -> str:
    """
    Executes web searches within the Routledge
    Encyclopedia of Philosophy (REP), a website
    containing summaries of professionally edited
    articles on a variety of topics in philosophy.
    Works best when queries are limited to a few
    words.

    Arguments:
        query: the search you would like to execute
    """

    #test comment

    web_search = DuckDuckGoSearchRun().run(
        f"site:https://www.rep.routledge.com {query}")
    return web_search
 
