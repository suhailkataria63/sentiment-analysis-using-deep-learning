library(shiny)
library(ggplot2)
library(dplyr)
library(readr)
library(httr)
library(jsonlite)

# Use the Shiny app directory as anchor, then go one level up to project root
# ---- Paths (final) ----
PROJECT_ROOT <- normalizePath("..", winslash = "/", mustWork = TRUE)
HIST_PATH <- file.path(PROJECT_ROOT, "data/processed/predictions/dl_history.csv")




# ---- Helpers ----
safe_read_csv <- function(path) {
  if (!file.exists(path)) return(NULL)
  readr::read_csv(path, show_col_types = FALSE)
}

call_predict_api <- function(text, api_base = "http://127.0.0.1:8001") {
  api_base <- sub("/+$", "", api_base)   # remove trailing slashes
  url <- paste0(api_base, "/predict")

  message("Calling URL: ", url)

  resp <- httr::POST(
    url,
    httr::add_headers(`Content-Type` = "application/json"),
    body = jsonlite::toJSON(list(text = text), auto_unbox = TRUE)
  )

  if (httr::status_code(resp) != 200) {
    return(list(error = TRUE, status = httr::status_code(resp), body = httr::content(resp, "text")))
  }

  data <- httr::content(resp, "parsed", simplifyVector = TRUE)
  data$error <- FALSE
  data
}
api_img_url <- function(path, api_base = "http://127.0.0.1:8001") {
  paste0(api_base, path)
}



message("BOOT getwd(): ", getwd())
message("BOOT PROJECT_ROOT: ", PROJECT_ROOT)
message("BOOT HIST_PATH: ", HIST_PATH)
message("BOOT HIST exists? ", file.exists(HIST_PATH))

# ---- UI ----
ui <- fluidPage(
  titlePanel("SocialMedia Intel Dashboard (R + DL + API)"),
  tabsetPanel(
    tabPanel("Overview",
      h4("Project Status"),
      tags$ul(
        tags$li("Spark pipeline produced Parquet + summaries"),
        tags$li("GRU sentiment model trained and saved"),
        tags$li("FastAPI serving predictions (/predict and /predict_batch)"),
        tags$li("This Shiny dashboard visualizes metrics and calls the API live")
      ),
      hr(),
      h4("Quick Links (local)"),
      tags$ul(
        tags$li(tags$a(href="http://127.0.0.1:8001/docs", "FastAPI Swagger UI (/docs)", target="_blank")),
        tags$li(tags$a(href="http://127.0.0.1:8001/", "FastAPI Web UI (/)", target="_blank"))
      )
    ),

    tabPanel("DL Metrics",
      h4("Training History (Accuracy / Loss)"),
      uiOutput("hist_status"),
      plotOutput("plot_acc"),
      plotOutput("plot_loss")
    ),

    tabPanel("Live Sentiment (API)",
      h4("Type text → Shiny calls FastAPI /predict"),
      textAreaInput("live_text", "Enter text:", value = "I love this product", rows = 4),
      actionButton("btn_predict", "Predict"),
      hr(),
      verbatimTextOutput("live_result")
    ),
    tabPanel("Data Metrics (EDA)",
  		h4("EDA Plots (served by FastAPI)"),
  		tags$p("These plots are generated from processed data and served as static PNGs via the API."),
 		 hr(),

  		h5("1) Sentiment Class Distribution"),
  		uiOutput("eda_status_class"),
  		uiOutput("eda_img_class"),
  		hr(),

  		h5("2) Text Length Distribution by Sentiment"),
  		uiOutput("eda_status_len"),
  		uiOutput("eda_img_len"),
  		hr(),

  		h5("3) Top Tokens by Sentiment"),
  		uiOutput("eda_status_tokens"),
  		uiOutput("eda_img_tokens")
	),

  )
)

tabPanel(
  "EDA Reports",
  selectInput(
    "eda_static_plot",
    "Choose plot",
    choices = c(
      "Class Distribution" = "class_distribution",
      "Text Length Distribution" = "text_length_distribution",
      "Top Tokens by Sentiment" = "top_tokens_by_sentiment"
    )
  ),
  uiOutput("eda_static_img")
)

# ---- Server ----
server <- function(input, output, session) {
    message("Shiny getwd(): ", getwd())
    message("PROJECT_ROOT: ", PROJECT_ROOT)
    message("HIST_PATH: ", HIST_PATH)
    message("HIST_PATH exists? ", file.exists(HIST_PATH))
    # --- EDA: Static plot URLs (FastAPI) ---
	EDA_CLASS_URL  <- api_img_url("/eda/static/class_distribution")
	EDA_LEN_URL    <- api_img_url("/eda/static/text_length_distribution")
	EDA_TOKENS_URL <- api_img_url("/eda/static/top_tokens_by_sentiment")

# Simple status text (so you know what URL is being used)
	output$eda_status_class <- renderUI({
  	tags$p(style="color:#555;", paste0("Source: ", EDA_CLASS_URL))
	})
	output$eda_status_len <- renderUI({
  	tags$p(style="color:#555;", paste0("Source: ", EDA_LEN_URL))
	})
	output$eda_status_tokens <- renderUI({
  	tags$p(style="color:#555;", paste0("Source: ", EDA_TOKENS_URL))
	})

# Render images
	output$eda_img_class <- renderUI({
  	tags$img(src = EDA_CLASS_URL, style="max-width:90%;
    width:750px;
    height:auto;
    display:block;
    margin-left:auto;
    margin-right:auto;
    border:1px solid #ddd;")
	})
	output$eda_img_len <- renderUI({
  	tags$img(src = EDA_LEN_URL, style="max-width:85%;
  	width:750px;
    height:auto;
    display:block;
    margin-left:auto;
    margin-right:auto;
    border:1px solid #ddd;")
	})
	output$eda_img_tokens <- renderUI({
  	tags$img(src = EDA_TOKENS_URL, style="max-width:85%;
  	width:750px;
    height:auto;
    display:block;
    margin-left:auto;
    margin-right:auto;
    border:1px solid #ddd;")
	})


  hist <- reactive({
    safe_read_csv(HIST_PATH)
  })

  output$hist_status <- renderUI({
    h <- hist()
    if (is.null(h)) {
      tags$p(style="color: #b00;",
             paste0("Not found: ", HIST_PATH, "  (Train the model / ensure dl_history.csv exists)"))
    } else {
      tags$p(style="color: #060;", paste0("Loaded: ", HIST_PATH, "  (epochs: ", nrow(h), ")"))
    }
  })
  output$eda_static_img <- renderUI({
    src <- paste0(
      "http://127.0.0.1:8001/eda/static/",
      input$eda_static_plot
    )
    tags$img(src = src, style = "max-width:100%;")
  })

  output$plot_acc <- renderPlot({
    h <- hist()
    if (is.null(h)) return(NULL)

    # Expected columns: accuracy, val_accuracy
    if (!("accuracy" %in% names(h)) || !("val_accuracy" %in% names(h))) return(NULL)

    h$epoch <- seq_len(nrow(h))
    ggplot(h, aes(x=epoch)) +
      geom_line(aes(y=accuracy, group=1)) +
      geom_line(aes(y=val_accuracy, group=1), linetype="dashed") +
      labs(title="Accuracy: Train vs Validation", y="Accuracy", x="Epoch") +
      theme_minimal()
  })

  output$plot_loss <- renderPlot({
    h <- hist()
    if (is.null(h)) return(NULL)

    # Expected columns: loss, val_loss
    if (!("loss" %in% names(h)) || !("val_loss" %in% names(h))) return(NULL)

    h$epoch <- seq_len(nrow(h))
    ggplot(h, aes(x=epoch)) +
      geom_line(aes(y=loss, group=1)) +
      geom_line(aes(y=val_loss, group=1), linetype="dashed") +
      labs(title="Loss: Train vs Validation", y="Loss", x="Epoch") +
      theme_minimal()
  })

  observeEvent(input$btn_predict, {
    txt <- input$live_text
    if (is.null(txt) || nchar(trimws(txt)) == 0) {
      output$live_result <- renderText("Please enter some text.")
      return()
    }

    # IMPORTANT: FastAPI must be running on 127.0.0.1:8000
    res <- call_predict_api(txt)

    if (isTRUE(res$error)) {
      output$live_result <- renderText(paste(
        "API call failed.",
        "\nStatus:", res$status,
        "\nBody:", res$body
      ))
      return()
    }

    output$live_result <- renderText(paste0(
      "Sentiment: ", res$sentiment, "\n",
      "Confidence: ", res$confidence, "\n",
      "P(positive): ", sprintf("%.4f", res$prob_positive), "\n",
      "Label: ", res$label, "\n\n",
      "Text: ", res$text
    ))
  })
}

shinyApp(ui, server)
