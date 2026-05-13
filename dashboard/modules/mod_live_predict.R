library(shiny)
library(httr)
library(jsonlite)

mod_live_predict_ui <- function(id) {
  ns <- NS(id)
  tagList(
    h3("Live Sentiment (API)"),
    textAreaInput(ns("text"), "Enter text:", value = "I love this product", rows = 4),
    actionButton(ns("go"), "Predict"),
    br(), br(),
    verbatimTextOutput(ns("result"))
  )
}

mod_live_predict_server <- function(id, api_base = "http://127.0.0.1:8000") {
  moduleServer(id, function(input, output, session) {

    output$result <- renderText("Click Predict to call the API...")

    observeEvent(input$go, {
      txt <- input$text
      if (is.null(txt) || nchar(trimws(txt)) == 0) {
        output$result <- renderText("Please enter some text.")
        return()
      }

      url <- paste0(api_base, "/predict")
      resp <- POST(
        url,
        add_headers(`Content-Type` = "application/json"),
        body = toJSON(list(text = txt), auto_unbox = TRUE)
      )

      if (status_code(resp) != 200) {
        output$result <- renderText(paste("API error:", status_code(resp), content(resp, "text")))
        return()
      }

      data <- content(resp, "parsed", simplifyVector = TRUE)

      out <- paste0(
        "Sentiment: ", data$sentiment, "\n",
        "Confidence: ", data$confidence, "\n",
        "P(positive): ", sprintf("%.4f", data$prob_positive), "\n",
        "Label: ", data$label, "\n\n",
        "Text: ", data$text
      )

      output$result <- renderText(out)
    })
  })
}
