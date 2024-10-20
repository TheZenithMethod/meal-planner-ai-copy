from fasthtml.common import *
from shad4fast import *
from lucide_fasthtml import Lucide

def layout(*args, **kwargs):
    """Custom layout for the Calorie Calculator app"""
    return Main()(
        Head(
            ShadHead(theme_handle=True),
        ),
        Body(cls="min-h-screen bg-background font-sans antialiased overflow-x-hidden")(
            Div(cls="relative flex min-h-screen flex-col")(
                Header(cls="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60")(
                    Div(cls="container flex h-14 items-center")(
                        MainNav(),
                        Div(cls="flex flex-1 items-center justify-between space-x-2 md:justify-end")(
                            ThemeToggle(),
                        )
                    )
                ),
                Div(cls="flex-1 overflow-x-hidden")(
                    Div(cls="container py-6")(
                        *args, **kwargs
                    )
                ),
                SiteFooter()
            )
        )
    )

def MainNav():
    return Nav(cls="flex items-center space-x-4 lg:space-x-6")(
        A(
            Img(src="https://challenge.thezenithmethod.com/wp-content/uploads/2024/01/ZenithLogo_01.svg", alt="Logo", cls="h-8 w-auto"),
            href="/",
            cls="flex items-center space-x-2"
        ),
        A("Calculate", href="/calculate", cls="text-sm font-medium transition-colors hover:text-primary"),
        A("About", href="/about", cls="text-sm font-medium text-muted-foreground transition-colors hover:text-primary")
    )

def ThemeToggle():
    return Button(
        Lucide("sun", cls="rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0"),
        Lucide("moon", cls="absolute rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100"),
        cls="relative h-9 w-9 rounded-md p-0 theme-toggle",
        variant="ghost",
        size="icon"
    )

def SiteFooter():
    return Footer(cls="border-t")(
        Div(cls="container flex flex-col items-center justify-between gap-4 py-10 md:h-24 md:flex-row md:py-0")(
            Div(cls="flex flex-col items-center gap-4 px-8 md:flex-row md:gap-2 md:px-0")(
                P("© 2024 Calorie Calculator. All rights reserved.", cls="text-center text-sm leading-loose text-muted-foreground md:text-left")
            )
        )
    )

def MainLayout(title, *content):
    return layout(
        H1(title, cls="text-3xl font-bold tracking-tight mb-4"),
        *content
    )
