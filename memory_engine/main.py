from fastapi import FastAPI

from memory_engine.api import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Memory Engine",
        description=(
            "A memory subsystem for conversational AI: atomic fact "
            "extraction, semantic retrieval, and safe, evaluated "
            "memory mutation."
        ),
    )
    app.include_router(router)
    return app


app = create_app()
