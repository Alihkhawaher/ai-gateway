import asyncio
from proxy import ProxyTUI, MODELS, get_config, load_config


async def main():
    load_config()
    app = ProxyTUI()
    async with app.run_test(size=(80, 50)) as pilot:
        await pilot.pause()

        # 1. Main screen should be active
        print("=== Main screen ===")
        print("current screen:", type(app.screen).__name__)
        log = app.screen.query_one("#log")
        print("log found:", log is not None)
        toolbar = app.screen.query_one("#main-toolbar")
        print("toolbar found:", toolbar is not None)

        # 2. Open settings via toolbar button
        print("\n=== Open Settings via toolbar button ===")
        await pilot.click("#menu-settings")
        await pilot.pause()
        print("current screen after click:", type(app.screen).__name__)

        # 3. Check settings widgets
        print("\n=== Settings screen widgets ===")
        for wid in ["model-select", "api-key-input", "host-input", "port-input", "addr-input",
                    "apply-btn", "back-btn", "restart-btn"]:
            try:
                w = app.screen.query_one(f"#{wid}")
                print(f"  {wid}: FOUND ({type(w).__name__})")
            except Exception as e:
                print(f"  {wid}: MISSING ({e})")

        # 4. Test Apply changes model
        print("\n=== Test Apply ===")
        print("  Before model:", MODELS[get_config("model_index")])
        settings_screen = app.screen
        select = settings_screen.query_one("#model-select")
        select.value = MODELS[1]
        await pilot.pause()
        settings_screen._apply_settings()
        await pilot.pause()
        print("  After model:", MODELS[get_config("model_index")])

        # 5. Back to main
        app.pop_screen()
        await pilot.pause()
        print("\n=== Back to main ===")
        print("current screen:", type(app.screen).__name__)

        app.exit()


asyncio.run(main())