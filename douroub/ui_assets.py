"""Visual assets for the app: the cedar watermark, the colour theme, and the trace-panel icons."""
import gradio as gr


# Cedar from the Lebanese flag: flat, layered branches with upturned tips. Used as a faint page watermark.
CEDAR_PATH = "M120,0 Q126,10 134,20 L106,20 Q114,10 120,0 Z M120,10 Q138.0,14.05 150,13.24 Q147.6,34 136.5,35 Q127.5,39 120,34 Q112.5,39 103.5,35 Q92.4,34 90,13.24 Q102.0,14.05 120,10 Z M120,32 Q152.4,36.05 174,35.24 Q169.68,56 149.7,57 Q133.5,61 120,56 Q106.5,61 90.3,57 Q70.32,56 66,35.24 Q87.6,36.05 120,32 Z M120,54 Q164.4,58.05 194,57.24 Q188.07999999999998,78 160.7,79 Q138.5,83 120,78 Q101.5,83 79.3,79 Q51.92,78 46,57.24 Q75.6,58.05 120,54 Z M120,76 Q174.0,80.05 210,79.24 Q202.8,100 169.5,101 Q142.5,105 120,100 Q97.5,105 70.5,101 Q37.2,100 30,79.24 Q66.0,80.05 120,76 Z M120,98 Q181.2,102.05 222,101.24 Q213.84,122 176.1,123 Q145.5,127 120,122 Q94.5,127 63.9,123 Q26.159999999999997,122 18,101.24 Q58.800000000000004,102.05 120,98 Z M120,120 Q186.0,124.05 230,123.24 Q221.2,144 180.5,145 Q147.5,149 120,144 Q92.5,149 59.49999999999999,145 Q18.799999999999997,144 10,123.24 Q54.0,124.05 120,120 Z M120,142 Q188.39999999999998,146.05 234,145.24 Q224.88,166 182.7,167 Q148.5,171 120,166 Q91.5,171 57.3,167 Q15.11999999999999,166 6,145.24 Q51.60000000000001,146.05 120,142 Z M112,166 L128,166 L131,208 L109,208 Z"

# Line icons for the trace panels, from Lucide (https://lucide.dev, ISC licence), coloured by CSS.
ICON_PATHS = {
    "search_experiences": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "compare_experiences": '<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M12 3v18"/>',
    "create_itinerary": ('<circle cx="6" cy="19" r="3"/><path d="M9 19h8.5a3.5 3.5 0 0 0 0-7h-11a3.5 3.5 0 0 1 0-7H15"/>'
                         '<circle cx="18" cy="5" r="3"/>'),
    "record_customer_interest": ('<rect width="20" height="16" x="2" y="4" rx="2"/>'
                                 '<path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/>'),
    "record_feedback": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "error": ('<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/>'
              '<path d="M12 9v4"/><path d="M12 17h.01"/>'),
    "approved": ('<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 '
                 '4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>'),
}

# Palette from the landscape: cedar green, sage stone.
CEDAR = gr.themes.Color(c50="#EEF5F1", c100="#D6E8DE", c200="#ADD0BE", c300="#7FB399", c400="#4F9374",
                        c500="#2F7658", c600="#245F47", c700="#1E4D3F", c800="#183E33", c900="#14332B",
                        c950="#0C211B", name="cedar")

STONE = gr.themes.Color(c50="#F6F8F6", c100="#EEF2EF", c200="#DCE3DE", c300="#C3CEC7", c400="#93A39A",
                        c500="#66786F", c600="#4B5D55", c700="#354640", c800="#233530", c900="#172823",
                        c950="#0E1A16", name="stone_green")

THEME = gr.themes.Soft(
    primary_hue=CEDAR, neutral_hue=STONE, radius_size="md",
    font=[gr.themes.GoogleFont("IBM Plex Sans Arabic"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#EEF2EC", body_background_fill_dark="#0E1A16",
    body_text_color="#14332B", body_text_color_dark="#E3ECE6",
    block_background_fill="#FFFFFF", block_background_fill_dark="#152520",
    block_border_color="#D7E1DA", block_border_color_dark="#253A33", block_shadow="none", block_radius="14px",
    background_fill_secondary_dark="#11201B", input_background_fill_dark="#1A2C26",
    color_accent_soft="#E2EDF2", color_accent_soft_dark="#1F3A46",
    button_primary_background_fill="#1E4D3F", button_primary_background_fill_hover="#245F47",
    button_primary_text_color="#FFFFFF", checkbox_background_color_selected="#1E4D3F",
    checkbox_border_color_selected="#1E4D3F", input_border_color_focus="#2F6E8F",
)

HEAD = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link href="https://fonts.googleapis.com/css2?family=Reem+Kufi:wght@500;600&display=swap" rel="stylesheet">')
