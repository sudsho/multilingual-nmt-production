variable "stage" {
  type    = string
  default = "staging"
}

variable "task_cpu" {
  type    = string
  default = "1024"
}

variable "task_mem" {
  type    = string
  default = "4096"
}

variable "desired_count" {
  type    = number
  default = 1
}
