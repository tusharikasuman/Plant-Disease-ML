from src import assistant as A, predict as P


def test_routes_and_answers_with_leaf(sample_images):
    m = P.load_models()
    p = P.predict(str(sample_images[0]), m)
    c = A.Context(prediction=p, models=m)
    assert A.ask("what disease is this?", c).intent == "diagnosis"
    assert A.ask("how do I treat late blight", c).intent == "treatment"
    assert "Random Forest" in A.ask("accuracy of random forest", c).text
    assert A.ask("", c).intent == "empty"
